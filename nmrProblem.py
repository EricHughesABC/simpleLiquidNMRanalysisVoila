# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 09:55:03 2021.

@author: ERIC
"""
import pandas as pd
import numpy as np
from numpy import pi, sin, cos, exp
from scipy import stats
from scipy import fftpack
import os
import nmrglue as ng
import yaml
import re

def read_in_cs_tables(h1:str, c13:str, scale_factor=6)->[pd.DataFrame, pd.DataFrame]:
    """

    Reads in json files for proton and chemical shift info and creates two
    pandas dataframes, one for proton and the other for carbon.


    Parameters.
    -----------
    h1 : str
        DESCRIPTION.
        filename of json file containing proton chemical shift information
    c13 : str
        DESCRIPTION.
        filename of json file containing proton chemical shift information
    scale_factor : TYPE, optional
        DESCRIPTION. The default is 6.

    Returns
    -------
    [pd.DataFrame, pd.DataFrame]

    """

    H1df = pd.read_json(h1)
    C13df = pd.read_json(c13)

    # create mean and sigma based on min max chemical shifts
    H1df['meanCS'] = (H1df.minCS + H1df.maxCS)/2
    H1df['sigmaCS'] = (H1df.maxCS - H1df.minCS)/scale_factor

    C13df['meanCS'] = (C13df.minCS + C13df.maxCS)/2
    C13df['sigmaCS'] = (C13df.maxCS - C13df.minCS)/scale_factor

    # create probability density functions for each chemical shitf group
    for i in H1df.index:
        H1df.loc[i, 'norm'] = stats.norm(loc=H1df.loc[i, 'meanCS'],
                                         scale=H1df.loc[i, 'sigmaCS'])

    for i in C13df.index:
        C13df.loc[i, 'norm'] = stats.norm(loc=C13df.loc[i, 'meanCS'],
                                          scale=C13df.loc[i, 'sigmaCS'])

    return H1df, C13df


class NMRproblem:

    def __init__(self, problemDirectory: str):
        """
        Creates a class to hold the particulars of the nmr problem.

        Parameters
        ----------
        problemDirectory : str
            DESCRIPTION. Holds path and directory of nmr problem

        Returns
        -------
        None.

        """

        self.problemDirectoryPath = problemDirectory
        self.rootDirectory, self.problemDirectory = os.path.split(problemDirectory)

        self.pngFiles = []
        self.jupyterFiles = []
        self.fidFilesDirectories = []
        self.yamlFiles = []
        self.excelFiles = []
        self.csvFiles = []
        self.pklFiles = []

        self.df = pd.DataFrame()
        self.dfBackup = pd.DataFrame()
        self.dfColumns = None
        self.dfIndex = None

        self.moleculePNGpanel = None
        self.moleculeAtomsStr = ""

        self.protonAtoms = []
        self.carbonAtoms = []
        self.elements = {}
        self.iprobs = {}

        self.numProtonGroups = 0
        self.numCarbonGroups = 0

        self.udic = {'ndim': 2,

                     0: {'obs': 400.0,
                         'sw': 12 * 400.0,
                         'dw': 1.0/(12 * 400.0),
                         'car': 12 * 400.0 / 2.0,
                         'size': int(32 * 1024),
                         'label': '1H',
                         'complex': True,
                         'encoding': 'direct',
                         'time': False,
                         'freq': True,
                         'lb': 0.5
                         },

                     1: {'obs': 100.0,
                         'sw': 210 * 100.0,
                         'dw': 1.0/(210 * 100.0),
                         'car': 210 * 100.0 / 2.0,
                         'size': int(1024*32),
                         'label': '13C',
                         'complex': True,
                         'encoding': 'direct',
                         'time': False,
                         'freq': True,
                         'lb': 0.5
                         }
                     }

        self.udic[0]['axis'] = ng.fileiobase.unit_conversion(self.udic[0]['size'],
                                                             self.udic[0]['complex'], 
                                                             self.udic[0]['sw'], 
                                                             self.udic[0]['obs'], 
                                                             self.udic[0]['car'])

        self.udic[1]['axis'] = ng.fileiobase.unit_conversion(self.udic[1]['size'],
                                                             self.udic[1]['complex'], 
                                                             self.udic[1]['sw'], 
                                                             self.udic[1]['obs'], 
                                                             self.udic[1]['car'])

        self.init_class_from_yml(problemDirectory)

    def init_class_from_yml(self, ymlFileNameDirName: str):
        """
        read in class parameters from yaml found in problem directory if found

        Parameters
        ----------
        ymlFileNameDirName : str
            DESCRIPTION. name and path to directory holding yaml file

        Returns
        -------
        bool
            DESCRIPTION. Return True if yaml found and processed

        """

        # Look for yaml file in directory and use first one found
        if os.path.isdir(ymlFileNameDirName):
            self.yamlFiles = [os.path.join(ymlFileNameDirName,f) for f in os.listdir( ymlFileNameDirName) if f.endswith('yml')]
            if len(self.yamlFiles) == 0:
                 return False
        elif ymlFileNameDirName.endswith('yml') and os.path.exists(ymlFileNameDirName):
            self.yamlFiles = [ymlFileNameDirName]
        else:
            return False

        if not os.path.exists(self.yamlFiles[0]):
            return False

        # open and read yaml file into dictionary and then process it
        with open(self.yamlFiles[0], 'r') as fp:
            info = yaml.safe_load(fp)
            self.init_variables_from_dict(info)

        return(True)


    def init_variables_from_dict(self, info: dict):
        """
        fill in class variables from dict produced by reading in yaml file.

        Parameters
        ----------
        info : dict
            DESCRIPTION. dictionary containing parameters that describe
            nmr problem.

        Returns
        -------
        None.

        """

        # columns and index  in best order for displaying df of parameters
        # describing problem
        self.dfColumns = info['dfColumns']
        self.dfIndex = info['dfIndex']

        # dataframe that has all the parameters that describe problem
        self.df = pd.DataFrame.from_dict(info['df'])
        if (len(self.dfColumns) > 0) and (len(self.dfIndex) > 0):
            self.df = self.df.loc[self.dfIndex, self.dfColumns]
        self.df_backup = self.df.copy()

        # parameters that describe the molecule in the problem and the number
        # of distinct peaks found in the carbon and proton NMR spectra
        self.moleculeAtomsStr = info.get('moleculeAtomsStr', self.moleculeAtomsStr )
        self.numProtonGroups = info.get('numProtonGroups', self.numProtonGroups)
        self.numCarbonGroups = info.get('numCarbonGroups', self.numCarbonGroups)
    
        # set of parameters that describe the spectrometer parameters
        # udic is based on nmrglue udic structure
        self.udic = info.get('udic', self.udic)

        # create nmr axes parameters from info in udic
        for i in range(self.udic['ndim']):        
            self.udic[i]['axis'] = ng.fileiobase.unit_conversion( self.udic[i]['size'],
                                                                  self.udic[i]['complex'], 
                                                                  self.udic[i]['sw'], 
                                                                  self.udic[i]['obs'], 
                                                                  self.udic[i]['car'])
            
        # create dbe and elements from moleculeAtomsStr
        self.calculate_dbe()

        # fill in protonAtoms and carbonAtoms list
        self.protonAtoms = [ 'H'+str(i+1) for i in range(self.numProtonGroups)]
        self.carbonAtoms = [ 'C'+str(i+1) for i in range(self.numCarbonGroups)]

        # put the list ofproton and carbon atoms in the udic for easier access
        self.udic[0]['atoms'] = self.protonAtoms
        self.udic[1]['atoms'] = self.carbonAtoms



    def calculate_dbe(self):
        """
        calculate DBE value for molecule and create a dictionary of elements
        and numbers found in molecule string

        Returns
        -------
        None.

        """
        # dbe_elements = ('C','H','N','F','Cl','Br')
        # match Element and number Cl, C3, O6, H
        aaa = re.findall(r'[A-Z][a-z]?\d?\d?\d?', self.moleculeAtomsStr)
        # match Element Cl, C, H, N
        eee = re.findall(r'[A-Z][a-z]?', self.moleculeAtomsStr)

        # create dictionary  of elements and number of elements
        self.elements = {}
        dbe_value = 0
        
        for e, a in zip(eee, aaa):
            if len(a) > len(e):
                num = a[len(e):]
            else:
                num = '1'

            self.elements[e] = int(num)

        # calcluate DBE value formolecule
        if 'C' in self.elements:
            dbe_value = self.elements['C']
        if 'N' in self.elements:
            dbe_value += self.elements['N']/2
        for e in ['H', 'F', 'Cl', 'Br']:
            if e in self.elements:
                dbe_value -= self.elements[e]/2

        self.dbe = dbe_value + 1



    def dfToNumbers(self):
        """
        converts table contents from strings to floats and integers where appropriate

        Returns
        -------
        None.

        """
        self.df.loc['integral', self.protonAtoms + self.carbonAtoms] = self.df.loc['integral', self.protonAtoms + self.carbonAtoms].astype(int)
        self.df.loc['C13 hyb', self.protonAtoms + self.carbonAtoms] = self.df.loc['C13 hyb', self.protonAtoms + self.carbonAtoms].astype(int)
        self.df.loc['attached protons', self.carbonAtoms] = self.df.loc['attached protons', self.carbonAtoms].astype(int)
        self.df.loc['ppm', self.protonAtoms + self.carbonAtoms] = self.df.loc['ppm', self.protonAtoms + self.carbonAtoms].astype(float)
        self.df.loc[self.protonAtoms + self.carbonAtoms, 'ppm'] = self.df.loc[ self.protonAtoms + self.carbonAtoms, 'ppm'].astype(float)


    def updateDFtable(self, qdf: pd.DataFrame):
        """
        receives dataframe from widget and stores it in classs dataframe

        Parameters
        ----------
        qdf : pd.DataFrame
            DESCRIPTION. Dataframe that has been changed 

        Returns
        -------
        bool
            DESCRIPTION. Returns True or False if problems in converting 
            values in dataframes to the correct type ie floats, ints, etc

        """

        self.df_backup = self.df.copy()
        self.df = qdf.copy()

        df = self.df

        # copy ppm values in row to columns
        atoms = self.protonAtoms + self.carbonAtoms
        proton_atoms = self.protonAtoms
        carbon_atoms = self.carbonAtoms
        df.loc[atoms, 'ppm'] = df.loc['ppm', atoms]

        # copy hmb and hsqc information over to carbon columns and proton rows
        for hi in proton_atoms:
            df.loc[hi, carbon_atoms] = df.loc[carbon_atoms, hi]

        hsqc = df.loc[proton_atoms, carbon_atoms]
        for hi in carbon_atoms:
            df.loc['hsqc', hi] = list(hsqc[hsqc[hi] == 'o'].index)
            df.loc['hmbc', hi] = list(hsqc[hsqc[hi] == 'x'].index)

        hsqc = df.loc[carbon_atoms, proton_atoms]
        for hi in proton_atoms:
            df.loc['hsqc', hi] = list(hsqc[hsqc[hi] == 'o'].index)
            df.loc['hmbc', hi] = list(hsqc[hsqc[hi] == 'x'].index)

        cosy = df.loc[proton_atoms]
        for hi in proton_atoms:
            df.loc['cosy', hi] = list(cosy[cosy[hi] == 'o'].index)
            
        return True

        # # turn string values to ints, floats and lists
        # try:
        #     #self.dfToNumbers()
        #    # convertHSQCHMBCCOSYtoLists(self)
        #    # convertJHzToLists(self)
            
        #     # qdf = df
            
            
        #     return True
        # except:
        #     df = self.df_backup.copy()
        #     return False

    def createInfoDataframes(self):
        """
        Initialize info dataframes from main df dataframe

        Returns
        -------
        None.

        """

        self.udic[0]['info'] = self.df.loc[['integral',
                                            'J type',
                                            'J Hz',
                                            'ppm',
                                            'cosy',
                                            'hsqc',
                                            'hmbc'], self.protonAtoms].T

        self.udic[0]['info']['labels'] = self.udic[0]['info'].index

        self.udic[1]['info'] = self.df.loc[['integral',
                                            'J type',
                                            'J Hz',
                                            'ppm',
                                            'cosy',
                                            'hsqc',
                                            'hmbc'], self.carbonAtoms].T

        self.udic[1]['info']['labels'] = self.udic[1]['info'].index

    def update_molecule_ipywidgets(self, molecule_str: str, pGrps: int, cGrps: int):
        """
        updates the molecule and number of proton and xcarbon groups observed
        in NMR spectra.
        If any changed a new NMR problem dataframe is started from scratch

        Parameters
        ----------
        molecule_str : str
            DESCRIPTION. string containing the number and type of atoms
        pGrps : int
            DESCRIPTION. number of distinct proton signals in NMR spectrum
        cGrps : int
            DESCRIPTION. number of distinct carbon signals in NMR spectrum

        Returns
        -------
        None.

        """

        changed = False
        if self.moleculeAtomsStr != molecule_str:
            changed = True
        if self.numProtonGroups != pGrps:
            changed = True
        if self.numCarbonGroups != cGrps:
            changed = True
        self.moleculeAtomsStr = molecule_str
        self.numProtonGroups = pGrps
        self.numCarbonGroups = cGrps

        self.calculate_dbe()

        # print(self.elements)

        if changed:
            # delete old dataframe and then recreate it with new params
            self.create_new_nmrproblem_df()
            
            
    def update_molecule(self, molecule_str: str, pGrps: int, cGrps: int)->bool:
        """
        updates the molecule and number of proton and xcarbon groups observed
        in NMR spectra.
        If any changed a new NMR problem dataframe is started from scratch

        Parameters
        ----------
        molecule_str : str
            DESCRIPTION. string containing the number and type of atoms
        pGrps : int
            DESCRIPTION. number of distinct proton signals in NMR spectrum
        cGrps : int
            DESCRIPTION. number of distinct carbon signals in NMR spectrum

        Returns
        -------
        bool
            DESCRIPTION. returns True if molecule or number of groups changed

        """

        changed = False
        if self.moleculeAtomsStr != molecule_str:
            changed = True
        if self.numProtonGroups != pGrps:
            changed = True
        if self.numCarbonGroups != cGrps:
            changed = True
        self.moleculeAtomsStr = molecule_str
        self.numProtonGroups = pGrps
        self.numCarbonGroups = cGrps

        self.calculate_dbe()

        # print(self.elements)

        return changed

    def create_new_nmrproblem_df(self):
        """
        Creates the dataframe template that holds the main info on the problem

        Returns
        -------
        None.

        """

        self.numProtonsInMolecule = self.elements['H']
        self.numCarbonsInMolecule = self.elements['C']

        self.protonAtoms = ['H'+str(i+1) for i in range(self.numProtonGroups)]
        self.carbonAtoms = ['C'+str(i+1) for i in range(self.numCarbonGroups)]

        self.dfIndex = ['integral',
                        'symmetry',
                        'symmetry factor',
                        'J type',
                        'J Hz',
                        'C13 hyb',
                        'attached protons',
                        'ppm'] \
                         + self.protonAtoms[::-1] \
                         + self.carbonAtoms[::-1] \
                         + ['hsqc', 'hmbc', 'cosy']
                        
        self.dfColumns = ['ppm'] + self.protonAtoms + self.carbonAtoms

        self.df = pd.DataFrame(index=self.dfIndex,
                               columns=self.dfColumns)

        self.df = self.df.fillna('')
        
        # update df with default values
        self.df.loc['integral', self.protonAtoms + self.carbonAtoms] = [1,] *len(self.protonAtoms + self.carbonAtoms)
        self.df.loc['J type', self.protonAtoms + self.carbonAtoms] = ['s',] *len(self.protonAtoms + self.carbonAtoms)
        self.df.loc['J Hz', self.protonAtoms + self.carbonAtoms] = ["[0]",] *len(self.protonAtoms + self.carbonAtoms)
        self.df.loc['hsqc', self.protonAtoms + self.carbonAtoms] = ["[]",] *len(self.protonAtoms + self.carbonAtoms)
        self.df.loc['hmbc', self.protonAtoms + self.carbonAtoms] = ["[]",] *len(self.protonAtoms + self.carbonAtoms)
        self.df.loc['cosy', self.protonAtoms ] = ["[]",] *len(self.protonAtoms)

        self.udic[0]['atoms'] = self.protonAtoms
        self.udic[1]['atoms'] = self.carbonAtoms

    def update_attachedprotons_c13hyb(self):

        for c in self.carbonAtoms:
            hsqc = self.df.loc['hsqc', c]
            attached_protons = 0
            for h in hsqc:
                attached_protons += int(self.df.loc['integral', h])

            self.df.loc['attached protons', c] = attached_protons
            self.df.loc['C13 hyb', c] = attached_protons

            for h in hsqc:
                self.df.loc['C13 hyb', h] = self.df.loc['C13 hyb', c]

    def extract_udic_base(self, udic):

        udicbasekeys = ['obs', 'sw', 'dw', 'car', 'size', 'label',
                        'complex', 'encoding', 'time', 'lb']

        udicbase = {}
        udicbase['ndim'] = udic['ndim']

        for i in range(udicbase['ndim']):
            udicbase[i] = {}
            for k in udicbasekeys:
                udicbase[i][k] = udic[i][k]

        return udicbase


    def save_problem(self):

        tobesaved = {}

        if hasattr(self, 'df'):
            if isinstance(self.df, pd.core.frame.DataFrame):
                tobesaved['df'] = self.df.to_dict()
                tobesaved['dfColumns'] = self.df.columns.tolist()
                tobesaved['dfIndex'] = self.df.index.tolist()

        if hasattr(self, 'moleculeAtomsStr'):
            tobesaved['moleculeAtomsStr'] = self.moleculeAtomsStr

        if hasattr(self, 'numProtonGroups'):
            tobesaved['numProtonGroups'] = self.numProtonGroups

        if hasattr(self, 'numCarbonGroups'):
            tobesaved['numCarbonGroups'] = self.numCarbonGroups

        tobesaved['udic'] = self.extract_udic_base(self.udic)

        return tobesaved

    def create1H13Clabels(self, num_poss=3):
       
        udic = self.udic
        iprobs = self.iprobs
        df = self.df
    
        for i in range(udic['ndim']):
            df = udic[i]['df']
            udic[i]['poss_subst_list'] = []
            for k, n in enumerate(udic[i]['atoms']):
                if num_poss == 1:
                    m = df.loc[iprobs[n][0], 'sF_latex_matplotlib']
                    sss = "{}: {}".format(n, m)
                    udic[i]['poss_subst_list'].append(sss)
                else:
                    output_list = [df.loc[l, 'sF_latex_matplotlib'] for l in iprobs[n][:2]]
                    sss = "{}$_{{{}}}$: ".format(n[0], n[1:])
                    sss += "\n".join(output_list)
                    udic[i]['poss_subst_list'].append(sss)

        for i in range(udic['ndim']):

            h1_info_zipped = zip(udic[i]['info'].labels.apply(str).tolist(),
                                 udic[i]['info'].ppm.apply(str).tolist(),
                                 udic[i]['info'].integral.apply(str).tolist())

            labels1 = ["{}$_{{{}}}$ {} ppm\nIntegral: {}".format(s[0][0], s[0][1], s[1], s[2]) for s in h1_info_zipped]

            h1_info_zipped = zip(udic[i]['info'].labels.apply(str).tolist(), 
                                 udic[i]['info'].ppm.apply(str).tolist(),
                                 udic[i]['info']['J type'].apply(str).tolist())

            labels2 = ["{}$_{{{}}}$ {} ppm\nJ type: {}".format(s[0][0], s[0][1], s[1], s[2]) for s in h1_info_zipped]

            labels3 = udic[i]['poss_subst_list']

            udic[i]['labels1_dict'] = {}
            for j, hi in enumerate(udic[i]['info'].index.tolist()):
                udic[i]['labels1_dict'][hi] = [labels3[j],labels2[j], labels1[j]]


    def calcProbDistFunctions(self, H1df_orig, C13df_orig):
        
        patoms = self.protonAtoms
        catoms = self.carbonAtoms
        
        # Figure out which dataframe has more rows
        if H1df_orig.index.size > C13df_orig.index.size:
            num_probs = H1df_orig.index.size
            iindex = H1df_orig.index
        else:
            num_probs = C13df_orig.index.size
            iindex = C13df_orig.index

        # create blank dataframe
        data = np.zeros((num_probs, len(patoms+catoms)))
        self.probDistFunctions = pd.DataFrame(data, 
                                              index = iindex, 
                                              columns=patoms + catoms)

        # Fill in probability density function table
        for H in patoms:
            for i in H1df_orig.index:
                ppm_val = float(self.df.loc[H,'ppm'])
                self.probDistFunctions.loc[i, H] = H1df_orig.loc[i, 'norm'].pdf(ppm_val)

        for C in catoms:
            for i in C13df_orig.index:
                ppm_val = float(self.df.loc[C,'ppm'])
                self.probDistFunctions.loc[i, C] = C13df_orig.loc[i, 'norm'].pdf(ppm_val)

        self.H1df = pd.concat([H1df_orig,
                               self.probDistFunctions[patoms]], axis=1)

        self.C13df = pd.concat([C13df_orig,
                                self.probDistFunctions[catoms]], axis=1)


    def identify1HC13peaks(self):

        M_substituents = ['Li', 'Na', 'K', 'Rb', 'Cs', 'Fr', 'Si', 'Al', 'B']

        elements = self.elements
        patoms = self.protonAtoms
        catoms = self.carbonAtoms
        df = self.df

        H1df = self.H1df
        C13df = self.C13df

        # reduce by DBE
        DBE = int(self.dbe)
        H1df = H1df[H1df['DBE'] <= DBE]
        C13df = C13df[C13df['DBE'] <= DBE]

        freeProtons = elements['H'] - df.loc['integral', patoms].sum()

        # If DBE equals 1, is it due to double bond oxygen, Nitrogen,
        # Halogen or alkene ?
        # carbonDBE = False
        oxygenDBE = False
        # halogenDBE = False
        # nitrogenDBE = False
        hydroxylsPresent = False
        numHydroxyls = 0
        freeOxygens = 0

        # Find out if there are hydroxyls
        freeProtons = elements['H'] - df.loc['integral', patoms].sum()
        if DBE == 0:
            if freeProtons > 0 and 'O' in elements:
                hydroxylsPresent = True
                numHydroxyls=freeProtons
                freeOxygens  = elements['O'] - freeProtons
                
        # Remove alkenes from table if DBE due to oxygen
        #
        # oxygenDBE = True
        # print(H1df.shape, C13df.shape)
        if ((DBE == 1) or (DBE >= 5 and elements['C'] > 6)) and (oxygenDBE):
            # remove alkene groups

            H1df = H1df[H1df.groupName != 'alkene']
            H1df = H1df[H1df.substituent != 'alkene']
            C13df = C13df[C13df.groupName != 'alkene']
            C13df = C13df[C13df.substituent != 'alkene']

        # remove choices where atoms not present, N, O, S, Halogens, metals
        # protons
        for e in ['O', 'S', 'N']:
            if e not in elements:
                H1df = H1df[H1df[e] == 0]
            else:
                H1df = H1df[H1df[e] <= elements[e]]

        no_halides = True

        halide_elements =[e for e in elements.keys() if e in ['F', 'Cl', 'Br', 'I']]
        if len(halide_elements)>0:
            no_halides = False

        if no_halides:
            H1df = H1df[H1df.substituent != 'halide']

        # remove metals from dataframe in none found
        no_Ms = True
        M_elements = [e for e in elements.keys() if e in M_substituents]
        if len(M_elements) > 0:
            no_Ms = False

        if no_Ms:
            H1df = H1df[H1df.substituent != 'metal']

        # carbons
        for e in ['O', 'S', 'N', 'F', 'Cl', 'Br', 'I']:
            if e not in elements:
                C13df = C13df[C13df[e] == 0]
            else:
                C13df = C13df[C13df[e] <= elements[e]]

        # Now to reduce the choice further by using integrals, multiplicity.
        # This should now be done for individual peaks and the results kept separately
        # start with carbons
        self.iprobs = {}
        for c in catoms:
            kept_i = []
            for i in C13df.index:
                if int(df.loc['C13 hyb', c]) in C13df.loc[i, 'attached_protons']:
                    kept_i.append(i)
            self.iprobs[c] = kept_i

        # if attached protons / hybr == 3 remove other guesses
        # other than C13 methyl
        for c in catoms:
            if df.loc['C13 hyb', c] == 3:
                kept_i = []
                for i in self.iprobs[c]:
                    # if first pos in list is 3 then group in table is a methyl
                    if C13df.loc[i, 'attached_protons'][0] == 3:
                        kept_i.append(i)
                self.iprobs[c] = kept_i

        # now reduce H1 options based on CH2, CH3, CH1
        # use the integral field in nmr properties table 
        for h in patoms:
            kept_i = []
            for i in H1df.index:
                if df.loc['C13 hyb', h] in H1df.loc[i, 'num_protons']:
                    kept_i.append(i)
            self.iprobs[h] = kept_i

        # check to see if proton is attached to a carbon
        #
        for h in patoms:
            kept_i = []
            for i in self.iprobs[h]:
                if (int(df.loc['C13 hyb', h]) > 0) and (H1df.loc[i, 'carbon_attached'] == 1):
                    kept_i.append(i)
                elif (int(df.loc['C13 hyb', h]) == 0) and (H1df.loc[i, 'carbon_attached'] == 0):
                    kept_i.append(i)
            self.iprobs[h] = kept_i

        # sort the candidates, highes first
        for n, iii in self.iprobs.items():
            if n in patoms:
                sss = H1df.loc[iii, n].sort_values(ascending=False)
            elif n in catoms:
                sss = C13df.loc[iii, n].sort_values(ascending=False)
            self.iprobs[n] = sss.index.tolist()

    def convertJHzToLists(self):

        df = self.df
        patoms = self.protonAtoms
        catoms = self.carbonAtoms

        for i, j in enumerate(df.loc['J Hz', patoms]):
            if isinstance(j,str):
                # print( [ float(k.strip()) for k in j.strip('][').split(',')])
                df.loc['J Hz', patoms[i]] = [float(k.strip()) for k in j.strip('][').split(',')]

        for i, j in enumerate(df.loc['J Hz', catoms]):
            if isinstance(j,str):
                # print( [ float(k.strip()) for k in j.strip('][').split(',')])
                df.loc['J Hz', catoms[i]] = [float(k.strip()) for k in j.strip('][').split(',')]
                


    def calculate1H13CSpectra1D(self):

        couplings = {'s': 0, 'd': 1, 't': 2, 'q': 3, 'Q': 4, 'S': 7}

        udic = self.udic

        for e in [0, 1]:  # [proton, carbon]
            expt = udic[e]
            npts = expt['size']
            lb = expt['lb']

            fid = np.zeros(npts, dtype=np.complex128)
            dw = expt['dw']
            ttt = np.linspace(0, dw * npts, npts)  # time array for calculating fid
            omega = expt['obs']  # Larmor freq in MHz
            centre_freq = expt['car']  # centre frequency in Hz

            expt['peak_ranges'] = {}

            for h1 in expt['atoms']:
                iso_ppm = expt['info'].loc[h1, 'ppm']  # isotropic chemical shift in ppm
                integral = expt['info'].loc[h1, 'integral']
                jType = expt['info'].loc[h1, 'J type']  # coupling string "dd" doublet of doublets
                jHz = expt['info'].loc[h1, 'J Hz']  # list of J coupling values in Hz

                # calculate isotropic fid for indivudual resonances
                isofreq = iso_ppm * omega - centre_freq  # isotropic chemical shift in Hz
                fid0 = integral * (cos(-2.0 * pi * isofreq * ttt) + 1j * sin(-2.0 * pi * isofreq * ttt))

                # add jcoupling modulation by iterating over string coupling values
                for i, j in enumerate(jType):
                    # print("i,j", i,j, type(j))
                    for k in range(couplings[j]):
                        # print("i,j, k", i,j, k)
                        fid0 = fid0 * cos(pi * jHz[i] * ttt)
                fid += fid0

                # fft individual peaks to define peak limits
                # starting from the beginning and end of the spectrum
                # we move inwards if the value is not five times bigger
                # than the baseline value at the extremes
                fid0 = fid0 * exp(-pi * lb * ttt)
                spec = fftpack.fftshift(fftpack.fft(fid0)).real

                base_line_value = spec[-1]

                ileft = 0
                iright = -1
                while spec[ileft] < 5*base_line_value:
                    ileft += 1
                while spec[iright] < 5*base_line_value:
                    iright -= 1

                expt['peak_ranges'][h1] = [ileft, npts + iright]

            # fft complete fid and store it
            fid = fid * exp(-pi * lb * ttt)
            expt['spec'] = fftpack.fftshift(fftpack.fft(fid)).real

            expt['spec'] = 0.9 * (expt['spec']/expt['spec'].max())

    def convertHSQCHMBCCOSYtoLists(self):
    #     global nmrproblem
        
        pcatoms = self.protonAtoms + self.carbonAtoms
        patoms = self.protonAtoms
        catoms = self.carbonAtoms
        df = self.df

        for i, j in enumerate(df.loc['hsqc', pcatoms]):
            if isinstance(j,str):
                # print( [ float(k.strip()) for k in j.strip('][').split(',')])
                df.loc['hsqc', pcatoms[i]] = [float(k.strip()) for k in j.strip('][').split(',')]

        for i, j in enumerate(df.loc['hmbc', pcatoms]):
            if isinstance(j,str):
                # print( [ k.strip() for k in j.strip('][').split(',')])
                df.loc['hmbc',pcatoms[i]] = [k.strip() for k in j.strip('][').split(',')]

        for i, j in enumerate(df.loc['cosy', patoms]):
            if isinstance(j,str):
                # print( [ float(k.strip()) for k in j.strip('][').split(',')])
                df.loc['cosy', patoms[i]] = [k.strip() for k in j.strip('][').split(',')]


    def populateInfoDict(self):

        hinfo = self.udic[0]['info']
        cinfo = self.udic[1]['info']
        
        hinfo['peak_ranges_pts'] = hinfo['peak_ranges'].values()
        cinfo['peak_ranges_pts'] = cinfo['peak_ranges'].values()
        
        udic = self.udic
        
        # make sure ppm column values are floats
        for i in range(udic['ndim']):
            udic[i]['info'].loc[udic[i]['atoms'], 'ppm'] = udic[i]['info'].ppm.astype(float)

        # make sure integrals values are ints
        for i in range(udic['ndim']):
            udic[i]['info'].loc[udic[i]['atoms'], 'integral'] = udic[i]['info'].integral.astype(int)

        # calculate peak ranges in ppm
        for i in range(udic['ndim']):
            udic[i]['peak_ranges_ppm'] = {}

            for k, v in udic[i]['peak_ranges'].items():
                udic[i]['peak_ranges_ppm'][k] = [udic[i]['axis'].ppm_scale()[udic[i]['peak_ranges'][k][0]],
                                                 udic[i]['axis'].ppm_scale()[udic[i]['peak_ranges'][k][1]]]

            udic[i]['info']['peak_ranges_ppm'] = udic[i]['peak_ranges_ppm'].values()

        # calculate peak positions in points and max height of peak
        for i in range(udic['ndim']):
            udic[i]['info']['pk_x'] = [-np.searchsorted(udic[i]['axis'].ppm_scale()[::-1],
                                        float(ppm)) for ppm in udic[i]['info'].ppm]

            udic[i]['info']['pk_y'] = [udic[i]['spec'][p[0]:p[1]].max() 
                                                  for p in udic[i]['info']['peak_ranges_pts']]
            
            
        # calculate peak widths in points starting from right hand side ie negative index
        for i in range(udic['ndim']):
            for j in udic[i]['atoms']:
                p = udic[i]['info'].loc[j, 'peak_ranges_pts']
                udic[i]['info'].loc[j,'pk_left'] = p[0]-udic[i]['size']
                udic[i]['info'].loc[j,'pk_right'] = p[1]-udic[i]['size']



    def save1DspecInfotoUdic(self):

        udic = self.udic

        udic[0]['info']['peak_ranges_pts'] = udic[0]['peak_ranges'].values()
        udic[1]['info']['peak_ranges_pts'] = udic[1]['peak_ranges'].values()

        for i in range(udic['ndim']):
            udic[i]['peak_ranges_ppm'] = {}

            for k, v in udic[i]['peak_ranges'].items():
                udic[i]['peak_ranges_ppm'][k] = [udic[i]['axis'].ppm_scale()[udic[i]['peak_ranges'][k][0]],
                                                 udic[i]['axis'].ppm_scale()[udic[i]['peak_ranges'][k][1]]]

            udic[i]['info']['peak_ranges_ppm'] = udic[i]['peak_ranges_ppm'].values()

        for i in range(udic['ndim']):
            udic[i]['info']['pk_x'] = [-np.searchsorted(udic[i]['axis'].ppm_scale()[::-1], float(ppm)) for ppm in udic[i]['info'].ppm]
            udic[i]['info']['pk_y'] = [udic[i]['spec'][p[0]:p[1]].max() for p in udic[i]['info']['peak_ranges_pts']]

        for i in range(udic['ndim']):
            for j in udic[i]['atoms']:
                p = udic[i]['info'].loc[j, 'peak_ranges_pts']
                udic[i]['info'].loc[j, 'pk_left'] = p[0]-udic[i]['size']
                udic[i]['info'].loc[j, 'pk_right'] = p[1]-udic[i]['size'] 