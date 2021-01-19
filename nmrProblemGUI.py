# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 14:17:17 2021

@author: ERIC
"""
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import mplcursors

import ipywidgets
from ipywidgets import widgets

import ipysheet
import nmrProblem

import numpy as np
import pandas as pd

import tempfile
import os
import yaml

import nmrglue as ng

# widget to hold pandas dataframe of NMR imformation extracted
# from spectra. The widget is based on a ipysheet widget and held in a VBOX
# so that we can update the widget when a dataset is read in or the molecule
# has been changed or a new problem is started.

# dfWarningTextW = widgets.Label("Table Messages:  OK")
# dfUpdateTableB = widgets.Button(description="update table")
# dfRunAnalysisB = widgets.Button(description="update and run")

# dfButtonsLayout = widgets.HBox([dfUpdateTableB,dfRunAnalysisB])
# dfLayout = widgets.VBox([dfWarningTextW])  # start with an empty widget
# dfLayout

h1 = r"csTables\h1_chemical_shift_table.jsn"
c13 = r"csTables\c13_chemical_shift_table.jsn"

H1df_orig, C13df_orig = nmrProblem.read_in_cs_tables(h1,c13)

class ipywidgetsDisplay(widgets.Tab):

    def __init__(self, nmrproblem=None):

        super().__init__()
        
        if not isinstance(nmrproblem, nmrProblem.NMRproblem):
            self.nmrproblem = nmrproblem
            self.df = pd.DataFrame()
        else:
            self.nmrproblem = nmrproblem
            self.df = nmrproblem.df

        # create debug label widget for output
        self.debugLabel = widgets.Label(value="",
                                        layout=widgets.Layout(width="400px"))

        # create save problem widgets
        self.saveProblemButtonW = widgets.Button(description="Save Problem")

        # widgets to obtain problem working directory
        self.prDirW = widgets.Text(value='',
                                   placeholder='problem directory',
                                   description='problem directory',
                                   disabled=False)

        self.prDirB = widgets.Button(description='Set Directory')

        self.upload_problemdir  = ipywidgets.widgets.FileUpload(multiple=True, 
                                                                description="Open Existing Problem ",
                                                                description_tooltip="choose all files in problem directory",
                                                                layout=widgets.Layout(width='300px'))


        self.problemNameL = widgets.Label(value="    Problem Name",
                                          layout=widgets.Layout(width='100px'))

        self.spacerL = widgets.Label(value="    ",
                                     layout=widgets.Layout(width='50px'))

        self.problemNameW = widgets.Text(value="Problem Name",
                                         description="",
                                         layout=widgets.Layout(width='150px'))

        self.newproblemB = widgets.Button(description="Start New Problem")

        self.prDirLayout = widgets.HBox([self.upload_problemdir,
                                         self.spacerL,
                                         self.problemNameL,
                                         self.problemNameW,
                                         self.spacerL,
                                         self.newproblemB])

        # widgets to obtain info on the molecule
        # number and tye of atoms in molecule
        # number of proton resonances in molecule
        # number of carbon resonance in molecule
        self.moleculeAtomsW = widgets.Text(value='',
                                           placeholder='atoms in molecule',
                                           description='atoms',
                                           disabled=False)

        self.pGrpsW = widgets.IntText(value=1,
                                      placeholder='H1 groups in spectrum',
                                      description='H1 groups',
                                      disabled=False)

        self.cGrpsW = widgets.IntText(value=1,
                                      description='C13 groups',
                                      disabled=False)

        self.moleculesSubmitB = widgets.Button(description="Update Molecule")

        self.moleculeLayout = widgets.VBox([self.moleculeAtomsW,
                                            self.pGrpsW,
                                            self.cGrpsW,
                                            self.moleculesSubmitB])

        # widgets to set 1D spectral parameters for proton and carbon
        self.pLabelW = widgets.Label("$^{1}H$")

        self.pSpecWidthW = widgets.FloatText(value=12.0,
                                             tooltip='proton spectral width',
                                             description='sw (ppm)',
                                             disabled=False)

        self.pObsFreqW = widgets.FloatText(value=400.0,
                                           description='obs (MHz)',
                                           disabled=False)

        self.pTofW = widgets.FloatText(value=5.0,
                                       description='tof (ppm)',
                                       diabled=False)

        self.pSizeW = widgets.IntText(value=32768,
                                      description='size (pts)',
                                      disabled=False)

        self.pLineBroadeningW = widgets.FloatText(value=0.5,
                                                  description='lb (Hz)',
                                                  disabled=False)

        self.cLabelW = widgets.Label("$^{13}C$")

        self.cSpecWidthW = widgets.FloatText(value=210.0,
                                             description='sw (ppm)',
                                             disabled=False)

        self.cObsFreqW = widgets.FloatText(value=100.0,
                                           description='obs (MHz)',
                                           disabled=False)

        self.cTofW = widgets.FloatText(value=5.0,
                                       description='tof (ppm)',
                                       diabled=False)

        self.cSizeW = widgets.IntText(value=32768,
                                      description='size (pts)',
                                      disabled=False)

        self.cLineBroadeningW = widgets.FloatText(value=0.5,
                                                  description='lb (Hz)',
                                                  disabled=False)

        self.specSubmitB = widgets.Button(description="Update Spectra")

        self.specLayout = widgets.HBox([widgets.VBox([self.pLabelW,
                                                      self.pObsFreqW,
                                                      self.pSpecWidthW,
                                                      self.pTofW,
                                                      self.pSizeW,
                                                      self.pLineBroadeningW,
                                                      self.specSubmitB]),
                                        widgets.VBox([self.cLabelW,
                                                      self.cObsFreqW,
                                                      self.cSpecWidthW,
                                                      self.cTofW,
                                                      self.cSizeW,
                                                      self.cLineBroadeningW])])

        self.old = 'All'
        self.new = 'ALL'

        self.toggleDF = widgets.ToggleButtons(options=['All',
                                                       'integrals-ppm',
                                                       'COSY',
                                                       'HSQC-HMBC'],
                                              description='Display:',
                                              disabled=False,
                                              button_style='',
                                              tooltips=['Show full Dataframe',
                                                        'Show COSY Input',
                                                        'Show HSQC/HMBC Input'])

        self.sheet1 = ipysheet.from_dataframe(self.df)

        self.toggleDF.observe(self.toggleValue)

        self.dfWarningTextW = widgets.Label("Table Messages:  OK")
        self.dfUpdateTableB = widgets.Button(description="update table")
        self.dfRunAnalysisB = widgets.Button(description="update and run")

        self.dfButtonsLayout = widgets.HBox([self.dfUpdateTableB,
                                             self.dfRunAnalysisB])

        self.dfLayout = widgets.VBox([self.toggleDF,
                                      self.dfWarningTextW,
                                      self.sheet1,
                                      self.dfButtonsLayout])

        self.accordion = widgets.Accordion(children=[self.prDirLayout,
                                                     self.moleculeLayout,
                                                     self.specLayout,
                                                     self.dfLayout])

        self.accordion.set_title(0, "Problem Directory")
        self.accordion.set_title(1, "Molecule")
        self.accordion.set_title(2, "Spectroscopy")
        self.accordion.set_title(3, "DataSet")
        self.page1 = widgets.VBox([self.accordion,
                                   self.saveProblemButtonW,
                                   self.debugLabel])

        self.H1C131DplotsLayout = widgets.VBox([widgets.Output(),
                                                self.saveProblemButtonW])

        self.ymlTitle = widgets.HTML("yml description of problem")
        self.ymlText = widgets.Textarea(layout=widgets.Layout(width="400px",
                                                              height="500px"))
        self.problemYML = widgets.VBox([self.ymlTitle,
                                        self.ymlText])

        self.children = [self.page1,
                         self.H1C131DplotsLayout,
                         self.problemYML]

        self.set_title(0,'Problem Setup')
        self.set_title(1, 'Problem Plots')
        self.set_title(2, 'Problem YML')

        self.upload_problemdir.observe(lambda change: self.on_upload_problemdir(change), names='value')
        self.moleculesSubmitB.on_click(self.onButtonClicked)
        self.specSubmitB.on_click(self.onButtonClicked)
        self.dfUpdateTableB.on_click(self.onButtonClicked)
        self.dfRunAnalysisB.on_click(self.onButtonClicked)
        self.saveProblemButtonW.on_click(self.onButtonClicked)
        self.newproblemB.on_click(self.onButtonClicked)


    def specwidthppm(self, udic, defaultvalue):
        sw = float(udic.get('sw', -1))
        if sw == -1:
            return defaultvalue
        else:
            return( sw / float(udic['obs']))
        
    def tofppm(self, udic, defaultvalue):
        tof = float(udic.get('car', -1))
        if tof == -1:
            return defaultvalue
        else:
            return( tof / float(udic['obs']))

    def updateSpectraWidgetsFromNMRproblem(self):
        
        nmrproblem = self.nmrproblem
    #     global nmrproblem
        # protons
        try:
            protons = nmrproblem.udic[0]
            self.pSpecWidthW.value = float(self.specwidthppm(protons, self.pSpecWidthW.value))
            self.pObsFreqW.value = float(protons.get('obs', self.pObsFreqW.value))
            self.pTofW.value = float(self.tofppm(protons, self.pTofW.value))
            self.pSizeW.value = int(protons.get('size', self.pSizeW.value))
            self.pLineBroadeningW.value = float(protons.get('lb', self.pLineBroadeningW.value))
        except:
            pass

        # carbons
        try:
            carbons = nmrproblem.udic[1]
            self.cSpecWidthW.value = float(self.specwidthppm(carbons, self.cSpecWidthW.value))
            self.cObsFreqW.value = float(carbons.get('obs', self.cObsFreqW.value))
            self.cTofW.value = float(self.tofppm(carbons, self.cTofW.value))
            self.cSizeW.value = int(carbons.get('size', self.cSizeW.value))
            self.cLineBroadeningW.value = float(carbons.get('lb', self.cLineBroadeningW.value))
        except:
            pass

    def display1H13C1Dspectra(self, ax):

        nmrproblem = self.nmrproblem

        udic = nmrproblem.udic

        if (1 in udic) and (0 in udic):
            if ('axis' in udic[1]) and ('spec' in udic[1]) and ('axis' in udic[1]) and ('spec' in udic[1]):
                full_spectrum1 = ax[0].plot(udic[1]['axis'].ppm_scale(), udic[1]['spec'], color='black', lw=0.5)
                ax[0].set_xlim(udic[1]['axis'].ppm_limits())
                full_spectrum2 = ax[1].plot(udic[0]['axis'].ppm_scale(), udic[0]['spec'], color='black', lw=0.5)
                ax[1].set_xlim(udic[0]['axis'].ppm_limits())

        # ax.set_xlim(10,-1)
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['left'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        ax[0].set_xlabel('$^{13}$C [ppm]', fontsize=10, gid='c13ppm')
        ax[0].set_yticks([])

        ax[1].spines['top'].set_visible(False)
        ax[1].spines['left'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].set_xlabel('$^{1}$H [ppm]', fontsize=10, gid='h1ppm')
        ax[1].set_yticks([])


    def plotC13Distributions(self, ax, numCandidates):
        
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('ppm', fontsize=10, gid='ppm')
        ax.set_yticks([])
        
        # plot top three candidates for each carbon present
        #
        C13_ppm_axis = np.linspace(-30,250,500)
        catoms = self.nmrproblem.carbonAtoms
        iprobs = self.nmrproblem.iprobs
        df = self.nmrproblem.df
        C13df = self.nmrproblem.udic[1]['df']

        c13distdict = {}

        for k, ci in enumerate(catoms):
            distlist = []
            for i in iprobs[ci][:numCandidates]:
                c13distr, = ax.plot(C13_ppm_axis,
                                    C13df.loc[i, 'norm'].pdf(C13_ppm_axis),
                                    label=C13df.loc[i, 'sF_latex_matplotlib'])

                c13distr.set_visible(False)
                distlist.append(c13distr)

            c13line = ax.axvline(float(df.loc['ppm', ci]))
            c13line.set_visible(False)
            distlist.append(c13line)

            c13distdict[ci] = distlist

        ax.set_xlabel('$^{13}$C [ppm]')
        ax.set_xlim(260, -40)
        return c13distdict


    def plotH1Distributions(self, ax, numCandidates):
        
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('ppm', fontsize=10, gid='ppm')
        ax.set_yticks([])

        # plot top three candidates for each carbon present
        #
        H1_ppm_axis = np.linspace(-5, 16, 500)
        patoms = self.nmrproblem.protonAtoms
        H1df = self.nmrproblem.udic[0]['df']
        iprobs = self.nmrproblem.iprobs
        df = self.nmrproblem.df

        h1distdict = {}

        for k, hi in enumerate(patoms):
            distlist = []
            for i in iprobs[hi][:numCandidates]:
                h1distr, = ax.plot(H1_ppm_axis,
                                   H1df.loc[i, 'norm'].pdf(H1_ppm_axis),
                                   label=H1df.loc[i, 'sF_latex_matplotlib'])

                h1distr.set_visible(False)
                distlist.append(h1distr)

            h1line = ax.axvline(float(df.loc['ppm', hi]))
            h1line.set_visible(False)
            distlist.append(h1line)

            h1distdict[hi] = distlist

        ax.set_xlabel('$^{1}$H [ppm]')
        ax.set_xlim(12, -2)

        return h1distdict



    def createH1C13interactivePlot(self): 

        w1 = widgets.Output()

        udic = self.nmrproblem.udic
        if 'info' not in udic[0]:
            return w1

        # delay plotting by using with statement
        # place matplotlib plot into a ipywidget
        with w1:
            # create four plots using a grid layout
            fig = plt.figure(constrained_layout=True, figsize=(9, 7))

            gs = GridSpec(2, 3, figure=fig)
            ax1 = fig.add_subplot(gs[0, :-1])
            ax2 = fig.add_subplot(gs[1, :-1])
            ax3 = fig.add_subplot(gs[0, -1])
            ax4 = fig.add_subplot(gs[1, -1])

            # put the axes into an array for easier axis
            ax = [ax1, ax2, ax3, ax4]

        # create 1D plots 1H and 13C in ax1 and ax2 respectively
        # carbon on top, proton on bottom
        self.display1H13C1Dspectra(ax)

        # plot all the distributions in iprobs and return each plot that belongs
        # to a specific label, C1, C2 .. in a dictionary of lists
        # the keys are the labels C1, C2, C3 or h1, H2, H3 ...
        # the values are a list of the indexes of the main df table that correspond to the peak based
        # on chemical shift, dbe, integrals

        c13distdict = self.plotC13Distributions(ax3, 3)

        h1distdict = self.plotH1Distributions(ax4, 3)

        # the dictionaries are put in a list so that they can be easier indexed
        h1c13distlist = [c13distdict, h1distdict]

        #  plot the peak overlays on top of the 1D 1H and 13C spectra
        # keep account of the different overlays by storing them in a dictionary of lists
        peak_overlays = []
        peak_overlays_dict = {}

        # for proton and carbon spectra
        for i in range(udic['ndim']):
            peak_overlays1 = []
            for Hi in udic[i]['info'].index:

                il = int(udic[i]['info'].loc[Hi, 'pk_left'])
                ir = int(udic[i]['info'].loc[Hi, 'pk_right'])

                pk, = ax[1-i].plot(udic[i]['axis'].ppm_scale()[il:ir],
                                   udic[i]['spec'][il:ir],
                                   lw=0.5,
                                   c='black',
                                   label=Hi,
                                   gid=Hi)

                peak_overlays1.append(pk)
                peak_overlays_dict[Hi] = pk

            peak_overlays.append(peak_overlays1)

        # connect the overlays to the cursor
        cursor = mplcursors.cursor(peak_overlays[0] + peak_overlays[1],
                                   hover=True,
                                   highlight=True)    

        # function to add the labels when the cursor hovers over a resonance.
        # cursor can be over carbon or proton 1D spectrum
        # if carbon proton connectivity is known via hsqc information 
        # then the coresponding
        # peak in the opposite spectrum is highlighed.
        # the probability distributions for the highlighted peak
        # are shown in the carbon and proton distribution plots
        @cursor.connect("add")
        def on_add(sel):

            if str(sel.artist.get_label()) in udic[0]['atoms']:
                ii = 0
            else:
                ii = 1
            x, y = sel.target

            # use artist to labal to find out peak id, H1, H2, ... or C1, C2
            lbl = sel.artist.get_label()
            selected_pk = str(lbl)

            # set selected peak to red with a linewidth of 0.75
            sel.extras[0].set_linewidth(0.75)
            sel.extras[0].set_color('red')

            # figure out if selected peak is a proton or a carbon
            if selected_pk in udic[0]['atoms']:
                ii = 0
            else:
                ii = 1

            # change selection text depending on what hight peak was picked
            # top, middle, bottom
            # pk_y is an index position from 0,1,2
            pk_y = 2-int(y*100/udic[ii]['info'].loc[lbl, 'pk_y'])//35

            # set annotation text
            sel.annotation.set_text(udic[ii]['labels1_dict'][selected_pk][pk_y])

            # highlight coresponding proton or carbon peaks
            # from hsqc data find corresponding peaks
            highlighted_pks = udic[ii]['info'].loc[selected_pk, 'hsqc']
            for hpk in highlighted_pks:
                sel.extras.append(cursor.add_highlight(peak_overlays_dict[hpk]))
                sel.extras[-1].set_linewidth(0.75)
                sel.extras[-1].set_color('red')

            # add highlighted distributions
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

            # used to dsplay legends of highlighted distributions
            plines1 = []
            plabels1 = []

            # set visible
            # circle through colors
            for hpk in highlighted_pks:
                for i, aa in enumerate(h1c13distlist[ii][hpk]):
                    plines1.append(aa)
                    plabels1.append(aa.get_label())

                    sel.extras.append(cursor.add_highlight(aa))
                    sel.extras[-1].set_visible(True)
                    sel.extras[-1].set_linewidth(0.75)
                    sel.extras[-1].set_color(colors[i % 7])

            # depending whether highlighted peak is carbon or proton
            # choose correct axis
            if ii == 0:
                ax[2].legend(plines1, plabels1)
            else:
                ax[3].legend(plines1, plabels1)

            # flip index for selected pk
            ii2 = 0
            if ii == 0:
                ii2 = 1

            # reset lists for legends
            plines1 = []
            plabels1 = []

            # set visible
            # circle through colors
            for i, aa in enumerate(h1c13distlist[ii2][selected_pk]):

                plines1.append(aa)
                plabels1.append(aa.get_label())

                sel.extras.append(cursor.add_highlight(aa))
                sel.extras[-1].set_visible(True)
                sel.extras[-1].set_linewidth(0.75)
                sel.extras[-1].set_color(colors[i % 3])

            # depending whether highlighted peak is carbon or proton
            # choose correct axis
            if ii == 0:
                ax[3].legend(plines1, plabels1)
            else:
                ax[2].legend(plines1, plabels1)

        return w1  # if everything works return plot in an ipywidget




    def on_upload_problemdir(self, change):

        files = change['new']
        file_names = list(files.keys())

        if len(file_names) == 0:
            return

        probdir_name, _ = file_names[0].split('.')
        print(probdir_name)

        with tempfile.TemporaryDirectory() as tmpdirname:

            print('created temporary directory', tmpdirname)
            problemDirectory = os.path.join(tmpdirname, probdir_name )
            os.mkdir(os.path.join(tmpdirname, probdir_name))

            for fname in files.keys():
                if "yml" in fname:
                    fp = open(os.path.join(problemDirectory, fname), 'wb')
                    fp.write(files[fname]["content"])
                    fp.close()

                elif "pkl" in fname:
                    fp = open(os.path.join(problemDirectory, fname), 'wb')
                    fp.write(files[fname]["content"])
                    fp.close()

            print(problemDirectory)
            self.nmrproblem = nmrProblem.NMRproblem(problemDirectory)

            if not isinstance(self.nmrproblem, type(None)):
                self.prDirW.value = self.nmrproblem.problemDirectoryPath

                # update other widgets based on contents of nmrproblem
                self.updateSpectraWidgetsFromNMRproblem()
                self.updateMoleculeWidgetsFromNMRproblem()

                # create a view of the dataframe in nmrproblem
                self.sheet1 = ipysheet.from_dataframe(self.nmrproblem.df)
                self.dfWarningTextW.value = "Table Messages: None"

                self.dfLayout.children = [self.toggleDF,
                                          self.dfWarningTextW,
                                          self.sheet1,
                                          self.dfButtonsLayout]

                # create 1D 1H & C13 plots widget
                self.H1C131DplotsLayout.children = [self.createH1C13interactivePlot(), self.saveProblemButtonW]


    def toggleValue(self, event):
        
        df = self.nmrproblem.df
        patoms = self.nmrproblem.protonAtoms
        catoms = self.nmrproblem.carbonAtoms

        if ipysheet.to_dataframe(self.sheet1).shape == (0, 0):
            return

        if event.name == 'value':

            self.old = event.old
            self.new = event.new
            # update backend dataframe

            # i
            if event.old == 'All':
                df.loc[:, :] = ipysheet.to_dataframe(self.sheet1).loc[:, :]

            elif event.old == 'COSY':
                rr = ['ppm'] + patoms
                cc = ['ppm'] + patoms
                df.loc[rr, cc] = ipysheet.to_dataframe(self.sheet1).loc[rr, cc]

            elif event.old == 'HSQC-HMBC':
                rr = ['ppm'] + catoms
                cc = ['ppm'] + patoms
                df.loc[rr, cc] = ipysheet.to_dataframe(self.sheet1).loc[rr, cc]

            elif event.old == 'integrals-ppm':
                rr = ['ppm', 'integral']
                cc = patoms + catoms

                df.loc[rr, cc] = ipysheet.to_dataframe(self.sheet1).loc[rr, cc]
                df.loc[cc, 'ppm'] = df.loc['ppm', cc]

            # update sheet1 display of dataframe
            if event.new == 'All':
                self.sheet1 = ipysheet.from_dataframe(df)

            elif event.new == 'COSY':
                rr = ['ppm'] + patoms[::-1]
                cc = ['ppm'] + patoms
                self.sheet1 = ipysheet.from_dataframe(df.loc[rr, cc])

            elif event.new == 'HSQC-HMBC':
                rr = ['ppm'] + catoms[::-1]
                cc = ['ppm'] + patoms
                self.sheet1 = ipysheet.from_dataframe(df.loc[rr, cc])

            elif event.new == 'integrals-ppm':
                rr = ['ppm', 'integral']
                cc = patoms + catoms

                self.sheet1 = ipysheet.from_dataframe(df.loc[rr, cc])

            self.dfLayout.children = [self.toggleDF,
                                      self.dfWarningTextW,
                                      self.sheet1,
                                      self.dfButtonsLayout]


    def onButtonClicked(self, bttn):

        self.debugLabel.value = bttn.description

    #     print("bttn", bttn, type(bttn), bttn.description)
        if "Start New" in bttn.description:
            self.debugLabel.value = self.problemNameW.value

            # remove spaces from name
            probdirname = self.problemNameW.value.replace(" ", "")

            tmpdir = tempfile.TemporaryDirectory()
            os.mkdir(os.path.join(tmpdir.name, probdirname))
            self.nmrproblem = nmrProblem.NMRproblem(os.path.join(tmpdir.name,
                                                                 probdirname))

            # create a temporary problem directory
        elif "Directory" in bttn.description:
            self.nmrproblem = nmrProblem.NMRproblem.from_guidata()
            if not isinstance( self.nmrproblem, type(None)):
                self.prDirW.value = self.nmrproblem.problemDirectoryPath

                # update other widgets based on contents of nmrproblem
                self.updateSpectraWidgetsFromNMRproblem()
                self.updateMoleculeWidgetsFromNMRproblem()

                # create a view of the dataframe in nmrproblem
                self.sheet1 = ipysheet.from_dataframe(self.nmrproblem.df)
                self.dfWarningTextW.value = "Table Messages: None"

#                 dfLayout.children = [dfWarningTextW, qgrid1, dfButtonsLayout]

                # create 1D 1H & C13 plots widget
                self.H1C131DplotsLayout.children = [ self.createH1C13interactivePlot() ]

        elif "update table" in bttn.description:
            print("update table")
            self.toggleDF.value = 'All'
            ok = self.nmrproblem.updateDFtable(ipysheet.to_dataframe(self.sheet1))

            if ok:
                self.nmrproblem.convertHSQCHMBCCOSYtoLists()
                self.nmrproblem.update_attachedprotons_c13hyb()
                self.sheet1 = ipysheet.from_dataframe(self.nmrproblem.df)
                self.dfWarningTextW.value = "Table Messages: None"
            else:
                 self.dfWarningTextW.value = "Table Messages: problems in table, please check it"

            self.dfLayout.children = [self.toggleDF,
                                      self.dfWarningTextW,
                                      self.sheet1,
                                      self.dfButtonsLayout]

        elif "update and run" in bttn.description:

            self.toggleDF.value='All'
            ok = self.nmrproblem.updateDFtable( ipysheet.to_dataframe(self.sheet1))

            if ok:
                self.sheet1 = ipysheet.from_dataframe(self.nmrproblem.df)
                self.dfWarningTextW.value = "Table Messages: None"

                self.nmrproblem.dfToNumbers()
                self.nmrproblem.convertHSQCHMBCCOSYtoLists()
                self.nmrproblem.convertJHzToLists()

                self.nmrproblem.update_attachedprotons_c13hyb()

#                 H1df_orig, C13df_orig = nmrProblem.readinChemShiftTables()
#                 H1df_orig, C13df_orig = read_in_cs_tables()

                self.nmrproblem.calcProbDistFunctions(H1df_orig, C13df_orig)

                self.nmrproblem.identify1HC13peaks()

                self.nmrproblem.udic[0]['df'] = self.nmrproblem.H1df
                self.nmrproblem.udic[1]['df'] = self.nmrproblem.C13df

                # self.nmrproblem.iprobs = iprobs

                self.nmrproblem.createInfoDataframes()

                self.nmrproblem.calculate1H13CSpectra1D()

                udic = self.nmrproblem.udic

                self.nmrproblem.save1DspecInfotoUdic()

                self.nmrproblem.create1H13Clabels(num_poss=3)

                self.dfLayout.children = [self.toggleDF,
                                          self.dfWarningTextW,
                                          self.sheet1,
                                          self.dfButtonsLayout]

                # create 1D 1H & C13 plots widget

                self.H1C131DplotsLayout.children = [self.createH1C13interactivePlot(), self.saveProblemButtonW]
            else:
                self.dfWarningTextW.value = "Table Messages: problems in table, please check it"

            self.dfLayout.children = [self.toggleDF,
                                      self.dfWarningTextW,
                                      self.sheet1,
                                      self.dfButtonsLayout]

        elif "Spectra" in bttn.description:
            self.nmrproblem.createInfoDataframes()
            self.nmrproblem.save1DspecInfotoUdic()

            self.updateSpectralInformationWidgetChanged()
            self.nmrproblem.calculate1H13CSpectra1D()
            self.H1C131DplotsLayout.children = [self.createH1C13interactivePlot(), self.saveProblemButtonW]

        elif "Molecule" in bttn.description:
            print("Molecule")

            self.nmrproblem.update_molecule(self.moleculeAtomsW.value,
                                            self.pGrpsW.value,
                                            self.cGrpsW.value)

            # create a view of the dataframe in nmrproblem
            self.sheet1 = ipysheet.from_dataframe(self.nmrproblem.df)
            self.dfWarningTextW.value = "Table Messages: None"

            self.dfLayout.children = [self.toggleDF,
                                      self.dfWarningTextW,
                                      self.sheet1,
                                      self.dfButtonsLayout]

        elif "Save Problem" in bttn.description:
            yml_dict = self.nmrproblem.save_problem()
            yml_str = yaml.dump(yml_dict, indent=4)
            self.ymlText.value = yml_str



    def updateSpectralInformationWidgetChanged(self):

        nmrproblem = self.nmrproblem

        # check to see if udic exists and proton and carbon axes are there
        if not hasattr(nmrproblem, 'udic'):
            return

        udic = nmrproblem.udic

        if 0 not in udic or 1 not in udic:
            return

        udic[0]['obs'] = self.pObsFreqW.value
        udic[0]['sw'] = self.pSpecWidthW.value * self.pObsFreqW.value
        udic[0]['dw'] = 1.0 / (self.pSpecWidthW.value * self.pObsFreqW.value)
        udic[0]['car'] = self.pTofW.value * self.pObsFreqW.value
        udic[0]['size'] = self.pSizeW.value
        udic[0]['lb'] = self.pLineBroadeningW.value
        udic[0]['complex'] = True

        udic[1]['obs'] = self.cObsFreqW.value
        udic[1]['sw'] = self.cSpecWidthW.value * self.cObsFreqW.value
        udic[1]['dw'] = 1.0 / (self.cSpecWidthW.value * self.cObsFreqW.value)
        udic[1]['car'] = self.cTofW.value * self.cObsFreqW.value
        udic[1]['size'] = self.cSizeW.value
        udic[1]['lb'] = self.cLineBroadeningW.value
        udic[1]['complex'] = True

        udic[0]['axis'] = ng.fileiobase.unit_conversion(udic[0]['size'],
                                                        udic[0]['complex'],
                                                        udic[0]['sw'],
                                                        udic[0]['obs'],
                                                        udic[0]['car'])

        udic[1]['axis'] = ng.fileiobase.unit_conversion(udic[1]['size'],
                                                        udic[1]['complex'],
                                                        udic[1]['sw'],
                                                        udic[1]['obs'],
                                                        udic[1]['car'])

    def updateMoleculeWidgetsFromNMRproblem(self):
        nmrproblem = self.nmrproblem
        self.moleculeAtomsW.value = nmrproblem.moleculeAtomsStr
        self.pGrpsW.value = nmrproblem.numProtonGroups
        self.cGrpsW.value = nmrproblem.numCarbonGroups
        
        
    # def create1H13C1Doverlays(self, ax):

    #     udic = self.nmrproblem.udic
    #     peak_overlays = []
    #     peak_overlays_dict = {}
    #     for i in range(udic['ndim']):
    #         peak_overlays1 = []
    #         for Hi in udic[i]['info'].index:
    #             il = int(udic[i]['info'].loc[Hi, 'pk_left'])
    #             ir = int(udic[i]['info'].loc[Hi, 'pk_right'])

    #             pk, = ax[1-i].plot(udic[i]['axis'].ppm_scale()[il:ir],
    #                                udic[i]['spec'][il:ir],
    #                                lw=0.5,
    #                                c='black',
    #                                label=Hi,
    #                                gid=Hi)

    #             peak_overlays1.append(pk)
    #             peak_overlays_dict[Hi] = pk

    #         peak_overlays.append(peak_overlays1)

    #     cursor = mplcursors.cursor(peak_overlays[0]+peak_overlays[1],
    #                                hover=True,
    #                                highlight=True)

    #     @cursor.connect("add")
    #     def on_add(sel):

    #         if str(sel.artist.get_label()) in udic[0]['atoms']:
    #             ii = 0
    #         else:
    #             ii = 1
    #         x, y = sel.target
    #         lbl = sel.artist.get_label()

    #         sel.extras[0].set_linewidth(0.75)
    #         sel.extras[0].set_color('red')
            
    #     return w1
