# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 13:50:51 2020

@author: vsmw51
"""
import os
import numpy as np
from numpy import pi, sin, cos, exp
import nmrglue as ng

from scipy import fftpack


import pandas as pd
import yaml
import re
import pickle

from matplotlib import pyplot as plt
import mplcursors

from ipywidgets import widgets

import guidata
import guidata.dataset.datatypes as dt
import guidata.dataset.dataitems as di




# class moleculeText(dt.DataSet):
#     """Molecule Formula"""
#     molecule_formula = di.StringItem("Enter Atoms and number present in molecule")

class moleculeText(dt.DataSet):
    """Molecule Formula"""
    reset_problem = di.BoolItem("Reset NMR problem", default=False)
    molecule_formula = di.StringItem("Enter Atoms and number present in molecule")
    protonGroupsInSpectrum = di.IntItem("Number of Proton Groups in Spectrum")
    carbonGroupsInSpectrum = di.IntItem("Number of Carbon Groups in Spectrum")
    
    
class problemDirectory(dt.DataSet):
    """NMR Directory"""
    probNameDir = di.DirectoryItem("NMR problem")


class NMRproblem:
    
    def __init__(self, problemDirectory):

        # self.problemDirectoryPath = None
        # self.rootDirectory = None
        # self.problemDirectory = None
        self.problemDirectoryPath = problemDirectory
        self.rootDirectory, self.problemDirectory = os.path.split(problemDirectory)
        
        self.pngFiles = []
        self.jupyterFiles = []
        self.fidFilesDirectories = []
        self.yamlFiles = []
        self.excelFiles = []
        self.csvFiles = []

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
                    0: { 'obs': 400.0,              
                        'sw': 12 * 400.0,
                        'dw': 1.0/(12 * 400.0),              
                        'car': 12 * 400.0 / 2.0,
                        'size': int(32*1024),
                        'label': '1H',
                        'complex': True,
                        'encoding': 'direct',
                        'time': False,
                        'freq': True,
                        'peak_ranges': {},
                        'lb': 0.5
                        },
                    
                    1: { 'obs': 100.0,              
                        'sw': 210 * 100.0,
                        'dw': 1.0/(210 * 100.0),              
                        'car': 210 * 100.0 / 2.0,
                        'size': int(1024*32),
                        'label': '13C',
                        'complex': True,
                        'encoding': 'direct',
                        'time': False,
                        'freq': True,
                        'peak_ranges': {},
                        'lb': 0.5
                        }
                    }

        self.udic[0]['axis'] = ng.fileiobase.unit_conversion( self.udic[0]['size'],
                                                              self.udic[0]['complex'], 
                                                              self.udic[0]['sw'], 
                                                              self.udic[0]['obs'], 
                                                              self.udic[0]['car'])

        self.udic[1]['axis'] = ng.fileiobase.unit_conversion( self.udic[1]['size'],
                                                              self.udic[1]['complex'], 
                                                              self.udic[1]['sw'], 
                                                              self.udic[1]['obs'], 
                                                              self.udic[1]['car'])
        
        self.molecule_gui = moleculeText()

        if not self.init_class_from_yml(problemDirectory):
            
            self.pngFiles = [f for f in os.listdir(problemDirectory) if f.endswith('png')]
            self.jupyterFiles = [f for f in os.listdir(problemDirectory) if f.endswith('ipynb')]
            self.fidFilesDirectories = [f for f in os.listdir(problemDirectory) if f.endswith('fid')]
            self.yamlFiles = [f for f in os.listdir(problemDirectory) if f.endswith('yml')]
            self.excelFiles = [f for f in os.listdir(problemDirectory) if f.endswith('xlsx')]
            self.csvFiles = [f for f in os.listdir(problemDirectory) if f.endswith('csv')]
            self.pklFiles = [f for f in os.listdir(problemDirectory) if f.endswith('pkl')]
            
            # self.problemDirectoryPath = problemDirectory
            # self.rootDirectory, self.problemDirectory = os.path.split(problemDirectory)
            
        elif self.pklFiles == []:
            # it might be that we have an old yml file so check directory
            # to see if there is a pkl file
            
            self.pklFiles = [f for f in os.listdir(problemDirectory) if f.endswith('pkl')]
            
        if len(self.pklFiles) > 0:
            #read in pkl data
            print(self.pklFiles)
            if self.problemDirectory in self.pklFiles[0]:
                print("read un pickle data")
                # try reading in pkl data
                fp = open(os.path.join(self.problemDirectoryPath, self.pklFiles[0]), 'rb')
                self.udic = pickle.load(fp)
                fp.close()
            
            
    # def save_to_yml(self, fpfn):
    #     pass
    
    def init_class_from_yml(self, ymlFileNameDirName):
        
        # print("Enter init_class_from_yml")
        
        if os.path.isdir(ymlFileNameDirName):
             self.yamlFiles = [os.path.join(ymlFileNameDirName,f) for f in os.listdir( ymlFileNameDirName) if f.endswith('yml')]
             if len(self.yamlFiles) == 0:
                 return False
        elif ymlFileNameDirName.endswith('yml') and os.path.exists(ymlFileNameDirName):
            self.yamlFiles = [ymlFileNameDirName]
        else:
            # print('return False 1')
            return False
        
        if not os.path.exists(self.yamlFiles[0]):
            # print(self.yamlFiles[0])
            # print('return False 1')
            return False
        
        with open(self.yamlFiles[0], 'r') as fp:
            info = yaml.safe_load(fp)
            
            # print(info)
                
        self.init_variables_from_dict(info)
            
        return(True)
    
    def init_variables_from_dict(self, info):
        
        # self.problemDirectoryPath = info['problemDirectoryPath']

        # self.problemDirectoryPath =info['problemDirectoryPath']
        # self.rootDirectory = info['rootDirectory']
        # self.problemDirectory = info['problemDirectory']
        
        self.pngFiles = info['pngFiles']
        self.jupyterFiles = info['jupyterFiles']
        self.fidFilesDirectories = info['fidFilesDirectories']
        self.yamlFiles = info['yamlFiles']
        self.excelFiles = info['excelFiles']
        self.csvFiles = info['csvFiles']
        self.pklFiles = info.get('pklFiles', [] )
        
        self.dfColumns = info['dfColumns']
        self.dfIndex = info['dfIndex']

        df = pd.DataFrame.from_dict(info['df'], 'index')
        self.df = df.loc[self.dfIndex, self.dfColumns]
        self.df_backup = self.df.copy()

        self.moleculeAtomsStr = info['moleculeAtomsStr']

        self.elements = info['elements']
        try:
            self.dbe = info['dbe']
        except:
            self.calculate_dbe()
            # print(self.dbe)

        self.protonAtoms = info['protonAtoms']
        self.carbonAtoms = info['carbonAtoms']

        self.numProtonGroups = info['numProtonGroups']
        self.numCarbonGroups = info['numCarbonGroups']
        self.udic[0]['atoms'] = info['protonAtoms']
        self.udic[1]['atoms'] = info['carbonAtoms']
        
        self.iprobs = info.get('iprobs', {})

    @classmethod
    def from_guidata(cls):

        probDir = problemDirectory()
        ok = probDir.edit()

        if ok:
            co = cls(probDir.probNameDir)
        else:
            co = None # cls(os.getcwd())

        return co

    def update_molecule_gui(self):
        self.molecule_gui = moleculeText()
        if  self.moleculeAtomsStr == "":
            self.molecule_gui.molecule_formula = ""
            self.molecule_gui.protonGroupsInSpectrum = 0
            self.molecule_gui.carbonGroupsInSpectrum = 0
        else:
            self.molecule_gui.molecule_formula = self.moleculeAtomsStr
            self.molecule_gui.protonGroupsInSpectrum = self.numProtonGroups
            self.molecule_gui.carbonGroupsInSpectrum = self.numCarbonGroups
    
        ok = self.molecule_gui.edit()

        if ok:
            changed = False
            if self.molecule_gui.reset_problem:
                changed = True
            if self.moleculeAtomsStr != self.molecule_gui.molecule_formula:
                changed = True
            if self.numProtonGroups != self.molecule_gui.protonGroupsInSpectrum:
                changed = True
            if self.numCarbonGroups != self.molecule_gui.carbonGroupsInSpectrum:
                changed = True
            self.moleculeAtomsStr = self.molecule_gui.molecule_formula
            self.numProtonGroups = self.molecule_gui.protonGroupsInSpectrum
            self.numCarbonGroups = self.molecule_gui.carbonGroupsInSpectrum

            self.calculate_dbe()
            
            # print(self.elements)

            if changed:
                # delete old dataframe and then recreate it with new params
                self.create_new_nmrproblem_df()


    def update_molecule_ipywidgets(self, molecule_str,pGrps,cGrps):



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

    def calculate_dbe(self):
        # dbe_elements = ('C','H','N','F','Cl','Br')
        # match Element and number Cl, C3, O6, H
        aaa =re.findall(r'[A-Z][a-z]?\d?\d?\d?',
                         self.moleculeAtomsStr)
        # match Element Cl, C, H, N
        eee = re.findall(r'[A-Z][a-z]?',
                           self.moleculeAtomsStr)

        # print("aaa", aaa)
        # print("eee", eee)

        # create dictionary  of elements and number of elements
    
        self.elements = {}
        dbe_value = 0
        
        for e, a in zip(eee, aaa):
            if len(a) > len(e):
                num = a[len(e):]
            else:
                num = '1'

            self.elements[e] = int(num)

        if 'C' in self.elements:
            dbe_value = self.elements['C']
        if 'N' in self.elements:
            dbe_value += self.elements['N']/2
        for e in ['H', 'F', 'Cl', 'Br']:
            if e in self.elements:
                dbe_value -= self.elements[e]/2

        self.dbe = dbe_value + 1

        # self.info['dbe'] = self.dbe
        # self.info['elements'] = self.elements

    # return int(dbe_value+1), elements

    def create_new_nmrproblem_df(self):
        
        # nmrProblem.info['molecular_formula'] = basics.molecule_formula
        # DBE, elements = calculate_dbe(nmrProblem.info['molecular_formula'])
        # print("DBE =", int(DBE))
        # print(elements)
        # nmrProblem.info['DBE']=DBE
        # nmrProblem.info['elements']=elements 
        
        # nmrProblem.info['numNMRobservedProtons'] = basics.numNMRobservedProtons
        # nmrProblem.info['numNMRobservedCarbons'] = basics.numNMRobservedCarbons
        
        self.numProtonsInMolecule = self.elements['H']
        self.numCarbonsInMolecule = self.elements['C']
        
        # print("self.numProtonGroups", self.numProtonGroups, type(self.numProtonGroups))
        # print("self.numCarbonGroups", self.numCarbonGroups, type(self.numCarbonGroups))

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
                          + ['hsqc',
                          'hmbc',
                          'cosy']
                        
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

    #     nmrProblem.info['df']=df

        # self.info['proton_atoms'] = self.proton_atoms
        # self.info['carbon_atoms'] = self.carbon_atoms
        # self.info['df_index'] = self.df_index
        # self.info['df_columns'] = self.df_columns

    def save_as_yml(self):
        info = {}

        # info['problemDirectoryPath'] = self.problemDirectoryPath

        # info['rootDirectory'] = self.rootDirectory
        # info['problemDirectory'] = self.problemDirectory
        info['pngFiles'] = self.pngFiles
        info['jupyterFiles'] = self.jupyterFiles
        info['fidFilesDirectories'] = self.fidFilesDirectories
        info['yamlFiles'] = self.fidFilesDirectories
        info['excelFiles'] = self.fidFilesDirectories
        info['csvFiles'] =  self.csvFiles
        info['pklFiles'] =  self.pklFiles
        
        info['dfColumns'] = self.dfColumns
        info['dfIndex'] = self.dfIndex
        
        info['df'] = self.df.to_dict('index')
        
        info['moleculeAtomsStr'] = self.moleculeAtomsStr

        info['elements'] = self.elements

        info['protonAtoms'] = self.protonAtoms
        info['carbonAtoms'] = self.carbonAtoms

        info['numProtonGroups'] = self.numProtonGroups
        info['numCarbonGroups'] = self.numCarbonGroups       

        info['yamlFiles'] = [self.problemDirectory + '.yml']
        
        info['iprobs'] = self.iprobs
        
        # print(os.path.join(self.problemDirectoryPath, self.problemDirectory + '.yml'))
        
        with open(os.path.join(self.problemDirectoryPath, self.problemDirectory + '.yml'), 'w') as fp:
            yaml.safe_dump(info, fp)
            
            
    def dfToNumbers(self):
        """converts table contents from strings to floats and integers where appropriate"""
        self.df.loc['integral', self.protonAtoms + self.carbonAtoms] = self.df.loc['integral', self.protonAtoms + self.carbonAtoms].astype(int)
        self.df.loc['C13 hyb', self.protonAtoms + self.carbonAtoms] = self.df.loc['C13 hyb', self.protonAtoms + self.carbonAtoms].astype(int)
        self.df.loc['attached protons', self.carbonAtoms] = self.df.loc['attached protons', self.carbonAtoms].astype(int)
        self.df.loc['ppm', self.protonAtoms + self.carbonAtoms] = self.df.loc['ppm', self.protonAtoms + self.carbonAtoms].astype(float)
        self.df.loc[self.protonAtoms + self.carbonAtoms, 'ppm'] = self.df.loc[ self.protonAtoms + self.carbonAtoms, 'ppm'].astype(float)

    def updateDFtable(self, qdf):

        # global nmrproblem
        
        # print("updateDFtable(nmrproblem, qdf)")
        
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
            df.loc[hi, carbon_atoms] =  df.loc[carbon_atoms, hi]
            
        hsqc = df.loc[proton_atoms,carbon_atoms]
        for hi in carbon_atoms:
            df.loc['hsqc',hi] = list(hsqc[hsqc[hi]=='o'].index)
            df.loc['hmbc',hi] = list(hsqc[hsqc[hi]=='x'].index)
        
        hsqc = df.loc[carbon_atoms,proton_atoms]
        for hi in proton_atoms:
            df.loc['hsqc',hi] = list(hsqc[hsqc[hi]=='o'].index)
            df.loc['hmbc',hi] = list(hsqc[hsqc[hi]=='x'].index)
        
        cosy = df.loc[proton_atoms]
        for hi in proton_atoms:
            df.loc['cosy',hi] = list(cosy[cosy[hi]=='o'].index)
        
        # turn string values to ints, floats and lists
        try:
            #self.dfToNumbers()
           # convertHSQCHMBCCOSYtoLists(self)
           # convertJHzToLists(self)
            
            # qdf = df
            
            
            return True
        except:
            df = self.df_backup.copy()
            return False



    def createInfoDataframes(self):
    #     global nmrproblem
        
        self.udic[0]['info'] = self.df.loc[['integral',
                                                        'J type',
                                                        'J Hz', 
                                                        'ppm', 
                                                        'cosy', 
                                                        'hsqc', 
                                                        'hmbc'], 
                                                        self.protonAtoms ].T

        self.udic[0]['info']['labels'] = self.udic[0]['info'].index
        
        self.udic[1]['info'] = self.df.loc[['integral',
                                                        'J type',
                                                        'J Hz', 
                                                        'ppm', 
                                                        'cosy', 
                                                        'hsqc', 
                                                        'hmbc'], 
                                                        self.carbonAtoms ].T

        self.udic[1]['info']['labels'] = self.udic[1]['info'].index            
    


# Create 1H and 13C labels for 1D spectra

def create1H13Clabels(nmrproblem, iprobs, num_poss=3):
   
    udic = nmrproblem.udic

    for i in range(udic['ndim']):
        df = udic[i]['df']
        udic[i]['poss_subst_list'] = []
        for k, n in enumerate(udic[i]['atoms']):
            if num_poss == 1:
                m = df.loc[iprobs[n][0], 'sF_latex_matplotlib']
                sss = "{}: {}".format(n, m)
                udic[i]['poss_subst_list'].append(sss)
            else:
                output_list = [df.loc[l,'sF_latex_matplotlib'] 
                                    for l in iprobs[n][:2] ]
                sss = "{}$_{{{}}}$: ".format(n[0],n[1:])
                sss += "\n".join(output_list)
                udic[i]['poss_subst_list'].append(sss)


    for i in range(udic['ndim']):
        
        h1_info_zipped = zip(udic[i]['info'].labels.apply(str).tolist(), 
                             udic[i]['info'].ppm.apply(str).tolist(),
                             udic[i]['info'].integral.apply(str).tolist())
    
        labels1 = ["{}$_{{{}}}$ {} ppm\nIntegral: {}".format(s[0][0], s[0][1], s[1], s[2]) for s in h1_info_zipped ]
    
        h1_info_zipped = zip(udic[i]['info'].labels.apply(str).tolist(), 
                             udic[i]['info'].ppm.apply(str).tolist(),
                             udic[i]['info']['J type'].apply(str).tolist())
    
        labels2 = ["{}$_{{{}}}$ {} ppm\nJ type: {}".format(s[0][0], s[0][1], s[1], s[2]) for s in h1_info_zipped ]
    
    
        labels3 = udic[i]['poss_subst_list']
    
        # print(labels1)
    
        udic[i]['labels1_dict'] = {}
        for j, hi in enumerate(udic[i]['info'].index.tolist()):
            udic[i]['labels1_dict'][hi] = [ labels3[j],labels2[j],labels1[j]]
            


def plotH1Distributions(nmrproblem, numCandidates=5):
    #plot top fice candidates for each proton
    
    H1_ppm_axis = np.linspace(-2,12.5,200)
    proton_atoms = nmrproblem.protonAtoms
    H1df = nmrproblem.udic[0]['df']
    
    fig, axes = plt.subplots(1,len(proton_atoms), figsize=(9,5))
    
    for k, n in enumerate(proton_atoms):
        for i in nmrproblem.iprobs[n][:numCandidates]:
            axes[k].plot(H1_ppm_axis, 
                H1df.loc[i,'norm'].pdf(H1_ppm_axis), 
                label= H1df.loc[i, 'sF_latex_matplotlib'])
                
        axes[k].legend();
        axes[k].set_title("{} = {} ppm".format(n,nmrproblem.df.loc['ppm',n]))
        axes[k].set_xlabel('$^1$H [ppm]')
        axes[k].set_xlim(12.5,0)
        axes[k].axvline(float(nmrproblem.df.loc['ppm',n]))

            

def plotC13Distributions(nmrproblem, numCandidates=5):
    
    # plot top three candidates for each carbon present
    #
    C13_ppm_axis = np.linspace(-30,250,500)
    carbon_atoms = nmrproblem.carbonAtoms
    C13df = nmrproblem.udic[1]['df']
    
    fig, axes = plt.subplots(1,len(carbon_atoms), figsize=(9,5))
    
    for k, n in enumerate(carbon_atoms):
        for i in nmrproblem.iprobs[n][:numCandidates]:
            axes[k].plot(C13_ppm_axis, C13df.loc[i,'norm'].pdf(C13_ppm_axis), 
                label= C13df.loc[i, 'sF_latex_matplotlib'])
            
        axes[k].axvline(float(nmrproblem.df.loc['ppm',n]))
    
        axes[k].legend();
        axes[k].set_title("{} = {} ppm".format(n,nmrproblem.df.loc['ppm',n]))
        axes[k].set_xlabel('$^{13}$C [ppm]')
        axes[k].set_xlim(260,-40)
        
    plt.show()
        
    #chemfigure


def calcProbDistFunctions(nmrproblem, H1df_orig, C13df_orig):
    
    proton_atoms = nmrproblem.protonAtoms
    carbon_atoms = nmrproblem.carbonAtoms
    
    # Figure out which dataframe has more rows
    if H1df_orig.index.size > C13df_orig.index.size:
        num_probs = H1df_orig.index.size
        iindex = H1df_orig.index
    else:
        num_probs = C13df_orig.index.size
        iindex = C13df_orig.index
    
    #create blank dataframe
    data = np.zeros((num_probs, len(proton_atoms+carbon_atoms)))
    probDistFunctions = pd.DataFrame(data, index = iindex, columns=proton_atoms+carbon_atoms)
    
    # Fill in probability density function table
    for H in proton_atoms:
        for i in H1df_orig.index:
            ppm_val = float(nmrproblem.df.loc[H,'ppm'])
            probDistFunctions.loc[i,H] = H1df_orig.loc[i,'norm'].pdf(ppm_val)
            
    for C in carbon_atoms:
        for i in C13df_orig.index:
            ppm_val = float(nmrproblem.df.loc[C,'ppm'])
            probDistFunctions.loc[i,C] = C13df_orig.loc[i,'norm'].pdf(ppm_val)
            
    H1df = pd.concat([H1df_orig, probDistFunctions[proton_atoms]], axis=1)
    C13df = pd.concat([C13df_orig, probDistFunctions[carbon_atoms]], axis=1)
    
    return H1df, C13df, probDistFunctions


# perform calculation of identifying 1H and 13C resonances

def identify1HC13peaks(nmrproblem, H1df, C13df):
    
    M_substituents = ['Li', 'Na', 'K', 'Rb', 'Cs', 'Fr', 'Si', 'Al', 'B']
    
    elements = nmrproblem.elements
    proton_atoms = nmrproblem.protonAtoms
    carbon_atoms = nmrproblem.carbonAtoms
    
    # reduce by DBE
    DBE = int(nmrproblem.dbe)
    H1df = H1df[H1df['DBE'] <= DBE]
    C13df = C13df[C13df['DBE'] <= DBE]
    
    freeProtons = elements['H'] - nmrproblem.df.loc['integral',proton_atoms].sum()
    
    # If DBE equals 1, is it due to double bond oxygen, Nitrogen, Halogen or alkene ?
    carbonDBE=False
    oxygenDBE=False
    halogenDBE=False
    nitrogenDBE=False
    hydroxylsPresent = False
    numHydroxyls = 0
    freeOxygens=0
    elements = nmrproblem.elements
    freeProtons = elements['H'] - nmrproblem.df.loc['integral',proton_atoms].sum()

    # Find out if there are hydroxyls
    freeProtons = nmrproblem.elements['H'] - nmrproblem.df.loc['integral',proton_atoms].sum()
    if DBE == 0:
        if freeProtons > 0 and 'O' in nmrproblem.elements:
            hydroxylsPresent = True
            numHydroxyls=freeProtons
            freeOxygens  = nmrproblem.elements['O'] - freeProtons
            
    # Remove alkenes from table if DBE due to oxygen
    #
    # oxygenDBE = True
    # print(H1df.shape, C13df.shape)
    if ((DBE == 1) or (DBE>=5 and elements['C'] > 6)) and (oxygenDBE):
        # remove alkene groups

        H1df = H1df[H1df.groupName != 'alkene']
        H1df = H1df[H1df.substituent != 'alkene']
        C13df = C13df[C13df.groupName != 'alkene']
        C13df = C13df[C13df.substituent != 'alkene']
    
    # remove choices where atoms not present, N, O, S, Halogens, metals
    ### protons

    for e in ['O','S','N']:
        if e not in elements:
            H1df=H1df[H1df[e]==0]
        else:
            H1df=H1df[H1df[e]<=elements[e]]

    no_halides = True

    halide_elements =[e for e in elements.keys() if e in ['F','Cl','Br', 'I']]
    if len(halide_elements)>0:
        no_halides = False

    if no_halides:
        H1df = H1df[H1df.substituent != 'halide']

    # remove metals from dataframe in none found
    no_Ms = True
    M_elements =[e for e in elements.keys() if e in M_substituents]
    if len(M_elements)>0:
        no_Ms = False

    if no_Ms:
        H1df = H1df[H1df.substituent != 'metal']

    ### carbons

    for e in ['O','S','N','F', 'Cl', 'Br', 'I']:
        if e not in elements:
            C13df = C13df[C13df[e] == 0]
        else:
            C13df = C13df[C13df[e]<=elements[e]]

    # Now to reduce the choice further by using integrals, multiplicity.
    # This should now be done for individual peaks and the results kept separately
    
    # start with carbons
    #
    iprobs = {}
    for c in carbon_atoms:
        kept_i = []
        for i in C13df.index:
            if int(nmrproblem.df.loc['C13 hyb', c]) in C13df.loc[i,'attached_protons']:
                kept_i.append(i)
        iprobs[c]=kept_i
        
    
        
    # if attached protons / hybr == 3 remove other guesses other than C13 methyl
    
    for c in carbon_atoms:
        if nmrproblem.df.loc['C13 hyb', c] == 3:
            kept_i = []
            for i in iprobs[c]:
                # if first pos in list is 3 then group in table is a methyl
                if C13df.loc[i,'attached_protons'][0] == 3:
                    kept_i.append(i)
            iprobs[c]=kept_i
    

    ## now reduce H1 options based on CH2, CH3, CH1
    ## use the integral field in nmr properties table 

    for h in proton_atoms:
        kept_i = []
        for i in H1df.index:
            if nmrproblem.df.loc['C13 hyb', h] in H1df.loc[i,'num_protons']:
                kept_i.append(i)
        iprobs[h]=kept_i

    # check to see if proton is attached to a carbon
    #
    for h in proton_atoms:
        kept_i = []
        for i in iprobs[h]:
            if (int(nmrproblem.df.loc['C13 hyb', h]) > 0) and (H1df.loc[i,'carbon_attached'] == 1):
                kept_i.append(i)
            elif (int(nmrproblem.df.loc['C13 hyb', h]) == 0) and (H1df.loc[i,'carbon_attached'] == 0):
                kept_i.append(i)
        iprobs[h]=kept_i

    # sort the candidates, highes first
    #
    for n, iii in iprobs.items():
        # print(n,iii)
#        sss = probDistFunctions.loc[iii, n].sort_values(ascending=False)
#        iprobs[n]=sss.index.tolist()
        if n in proton_atoms:
            sss = H1df.loc[iii, n].sort_values(ascending=False)
        elif n in carbon_atoms:
            sss = C13df.loc[iii, n].sort_values(ascending=False)
        iprobs[n]=sss.index.tolist()

    return iprobs


def readinChemShiftTables():
    
    H1df_fn = r"../book_examples/python_files/H1_chemical_shift_table.pkl"
    H1df_orig = pd.read_pickle(H1df_fn)
    
    C13df_fn = r"../book_examples/python_files/C13_chemical_shift_table.pkl"
    C13df_orig = pd.read_pickle(C13df_fn)
    
    return H1df_orig, C13df_orig


# create a widget to display the 1D 1H & C13 plots

def create1D_H1C13plot(nmrproblem):
    # set up plot

    spectrOutputW = widgets.Output()
    
    try:
        udic = nmrproblem.udic
        with spectrOutputW:
            fig, ax = plt.subplots(udic['ndim'],figsize=(9, 7))

        full_spectrum1 = ax[0].plot(udic[1]['axis'].ppm_scale(), udic[1]['spec'], color='black', lw=0.5)
        ax[0].set_xlim(udic[1]['axis'].ppm_limits())
        full_spectrum2 = ax[1].plot(udic[0]['axis'].ppm_scale(), udic[0]['spec'], color='black', lw=0.5)
        ax[1].set_xlim(udic[0]['axis'].ppm_limits())
        # ax.set_xlim(10,-1)
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['left'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        ax[0].set_xlabel('ppm', fontsize=12, gid='ppm')
        ax[0].set_yticks([])

        ax[1].spines['top'].set_visible(False)
        ax[1].spines['left'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].set_xlabel('ppm', fontsize=12, gid='ppm')
        ax[1].set_yticks([])

        peak_overlays = []
        peak_overlays_dict = {}

        for i in range(udic['ndim']):
            peak_overlays1 = []
            for Hi in udic[i]['info'].index:

                il = int(udic[i]['info'].loc[Hi,'pk_left'])
                ir = int(udic[i]['info'].loc[Hi,'pk_right'])


                pk, = ax[1-i].plot(udic[i]['axis'].ppm_scale()[il:ir], udic[i]['spec'][il:ir], 
                                      lw=0.5, c='black', label=Hi, gid=Hi)
                peak_overlays1.append(pk)
                peak_overlays_dict[Hi] = pk

            peak_overlays.append(peak_overlays1)

        pairs = {}
        for hi in udic[0]['atoms']:
            ci = udic[0]['info'].loc[hi,'hsqc']
            if len(ci) == 1:
                pairs[peak_overlays_dict[hi]] = peak_overlays_dict[ci[0]]
                pairs[peak_overlays_dict[ci[0]]] = peak_overlays_dict[hi]
        # pairs = dict(zip(peak_overlays[0], peak_overlays[1][1:]))
        # pairs.update(zip(peak_overlays[1][1:], peak_overlays[0]))    
        # Attach tool tip hints to graph
        cursor = mplcursors.cursor(peak_overlays[0]+peak_overlays[1],   hover= True, highlight=True)

        @cursor.connect("add")
        def on_add(sel):

        #             pass
            if str(sel.artist.get_label()) in udic[0]['atoms']:
                ii = 0
            else:
                ii = 1
            x,y = sel.target
            # print(x,y, udic[ii]['spec'].max())

            lbl = sel.artist.get_label()

            sel.extras[0].set_linewidth(0.75)
            sel.extras[0].set_color('red')

            pk_y = 2-int(y*100/udic[ii]['info'].loc[lbl,'pk_y'])//35
            sel.annotation.set_text(udic[ii]['labels1_dict'][str(sel.artist.get_label())][pk_y])
            if sel.artist in pairs:
                sel.extras.append(cursor.add_highlight(pairs[sel.artist]))
                sel.extras[-1].set_linewidth(0.75)
                sel.extras[-1].set_color('red')
        #         if ii==1:
        #             sel.extras.append(cursor.add_annotation.set_text(udic[0]['labels1_dict'][str(sel.artist.get_label())][pk_y]))
        #         else:
        #             sel.extras.append(cursor.add_annotation.set_text(udic[1]['labels1_dict'][str(sel.artist.get_label())][pk_y]))

        return spectrOutputW
    except:
        return spectrOutputW


def convertJHzToLists(nmrproblem):
#     global nmrproblem
    
    for i, j in enumerate(nmrproblem.df.loc['J Hz', nmrproblem.protonAtoms]):
        if isinstance(j,str):
            # print( [ float(k.strip()) for k in j.strip('][').split(',')])
            nmrproblem.df.loc['J Hz', nmrproblem.protonAtoms[i]] = [ float(k.strip()) for k in j.strip('][').split(',')]

    for i, j in enumerate(nmrproblem.df.loc['J Hz', nmrproblem.carbonAtoms]):
        if isinstance(j,str):
            # print( [ float(k.strip()) for k in j.strip('][').split(',')])
            nmrproblem.df.loc['J Hz', nmrproblem.carbonAtoms[i]] = [ float(k.strip()) for k in j.strip('][').split(',')]


def calculate1H13CSpectra1D(nmrproblem):

#     global nmrproblem
    

    couplings = {'s': 0, 'd': 1, 't': 2, 'q': 3, 'Q':4, 'S': 7}

    for e in [0,1]: # [proton, carbon]
        expt = nmrproblem.udic[e]
        npts = expt['size']
        lb = expt['lb']

        fid = np.zeros(npts, dtype=np.complex128)
        dw =expt['dw']
        ttt = np.linspace(0, dw * npts, npts) # time array for calculating fid
        omega = expt['obs'] # Larmor freq in MHz
        centre_freq = expt['car']  # centre frequency in Hz

        for h1 in expt['atoms']:
    #         info = expt['info']
            iso_ppm = expt['info'].loc[h1,'ppm']  # isotropic chemical shift in ppm
            integral = expt['info'].loc[h1, 'integral']
            jType = expt['info'].loc[h1, 'J type'] # coupling string "dd" doublet of doublets
            jHz = expt['info'].loc[h1, 'J Hz'] # list of J coupling values in Hz

            # calculate isotropic fid for indivudual resonances
            isofreq = iso_ppm * omega - centre_freq  # isotropic chemical shift in Hz
            fid0 = integral * (cos(-2.0 * pi * isofreq * ttt) + 1j * sin(-2.0 * pi * isofreq * ttt))
            
            # add jcoupling modulation by iterating over string coupling values
            for i,j in enumerate(jType):
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

            expt['peak_ranges'][h1]=[ileft,npts+iright]

        # fft complete fid and store it
        fid = fid * exp(-pi * lb * ttt)
        expt['spec'] = fftpack.fftshift(fftpack.fft(fid)).real

        expt['spec'] = 0.9 * (expt['spec']/expt['spec'].max())


def convertHSQCHMBCCOSYtoLists(nmrproblem):
#     global nmrproblem
    
    protonCarbonAtoms = nmrproblem.protonAtoms + nmrproblem.carbonAtoms
    
    for i, j in enumerate(nmrproblem.df.loc['hsqc', protonCarbonAtoms]):
        if isinstance(j,str):
            # print( [ float(k.strip()) for k in j.strip('][').split(',')])
            nmrproblem.df.loc['hsqc', protonCarbonAtoms[i]] = [ float(k.strip()) for k in j.strip('][').split(',')]

    for i, j in enumerate(nmrproblem.df.loc['hmbc', protonCarbonAtoms]):
        if isinstance(j,str):
            # print( [ k.strip() for k in j.strip('][').split(',')])
            nmrproblem.df.loc['hmbc', protonCarbonAtoms[i]] = [ k.strip() for k in j.strip('][').split(',')]

    for i, j in enumerate(nmrproblem.df.loc['cosy', nmrproblem.protonAtoms]):
        if isinstance(j,str):
            # print( [ float(k.strip()) for k in j.strip('][').split(',')])
            nmrproblem.df.loc['cosy', nmrproblem.protonAtoms[i]] = [ k.strip() for k in j.strip('][').split(',')]
            




def populateInfoDict(nmrproblem):
#    global nmrproblem
    
    hinfo = nmrproblem.udic[0]['info']
    cinfo = nmrproblem.udic[1]['info']
    
    hinfo['peak_ranges_pts'] = hinfo['peak_ranges'].values()
    cinfo['peak_ranges_pts'] = cinfo['peak_ranges'].values()
    
    udic = nmrproblem.udic
    
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



        
    
def save1DspecInfotoUdic(nmrproblem):
    
    udic = nmrproblem.udic

    udic[0]['info']['peak_ranges_pts'] = udic[0]['peak_ranges'].values()
    udic[1]['info']['peak_ranges_pts'] = udic[1]['peak_ranges'].values()
    
    for i in range(udic['ndim']):
        udic[i]['peak_ranges_ppm'] = {}

        for k, v in udic[i]['peak_ranges'].items():
#                     print(k)
            udic[i]['peak_ranges_ppm'][k] = [udic[i]['axis'].ppm_scale()[udic[i]['peak_ranges'][k][0]], 
                                             udic[i]['axis'].ppm_scale()[udic[i]['peak_ranges'][k][1]]]


        udic[i]['info']['peak_ranges_ppm'] = udic[i]['peak_ranges_ppm'].values()
        
    for i in range(udic['ndim']):
        udic[i]['info']['pk_x'] = [-np.searchsorted(udic[i]['axis'].ppm_scale()[::-1], float(ppm)) for ppm in udic[i]['info'].ppm]
        udic[i]['info']['pk_y'] = [udic[i]['spec'][p[0]:p[1]].max() for p in udic[i]['info']['peak_ranges_pts']]

        
    for i in range(udic['ndim']):
        for j in udic[i]['atoms']:
            p = udic[i]['info'].loc[j, 'peak_ranges_pts']
            udic[i]['info'].loc[j,'pk_left'] = p[0]-udic[i]['size']
            udic[i]['info'].loc[j,'pk_right'] = p[1]-udic[i]['size']    
    
if __name__ == "__main__":

    _app = guidata.qapplication()  # not required if a QApplication has already been created

    nmrproblem = NMRproblem.from_guidata()
    
    # print(type(nmrproblem))
    
    if isinstance(nmrproblem, NMRproblem):
    
        nmrproblem.update_molecule_gui()
        
        nmrproblem.save_as_yml()
        
        print(nmrproblem.df)
        
        H1df_orig, C13df_orig = readinChemShiftTables()
        
        H1df, C13df, probDistFunctions = calcProbDistFunctions(nmrproblem, H1df_orig, C13df_orig)
        
        iprobs = identify1HC13peaks(nmrproblem, H1df, C13df)
        
        nmrproblem.udic[0]['df'] = H1df
        nmrproblem.udic[1]['df'] = C13df
        
        create1H13Clabels(nmrproblem, iprobs, num_poss=3)
        
        plotC13Distributions(nmrproblem)
        plotH1Distributions(nmrproblem, numCandidates=5)
        
        www = create1D_H1C13plot(nmrproblem)


