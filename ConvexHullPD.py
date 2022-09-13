# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

import plotly.graph_objects as go
import logging

from chemicalDiagram.EquilibriumLine import EquilLine

import re


from scipy.spatial import ConvexHull, HalfspaceIntersection
import numpy as np


try:
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb

from pymatgen.core.composition import Composition

from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.analysis.reaction_calculator import ComputedReaction
import os
import json
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDPlotter
from myResearch.getOrigStableEntriesList import getOrigStableEntriesList
# MPR = MPRester("2d5wyVmhDCpPMAkq")
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

__author__ = "Jiadong Chen"
__copyright__ = "Copyright 2020"
__version__ = "0.1"
__maintainer__ = "Jiadong Chen"
__credits__ = "Jiadong Chen"
__email__ = "jiadongc@umich.edu"
__status__ = "Production"
__date__ = "June 12th, 2020"

logger = logging.getLogger(__name__)

MU_H2O = -2.4583
PREFAC = 0.0591



class Convex_hull(EquilLine):
    """
    Class to create a Equilibrium hyperplane on 
    chemical potential diagram from CPentries

    Args:
        entries (CPentries): Entries list
            containing Solids and Ions or a list of MultiEntries
        elementList: element str list of entries
        fixed:  list of tuples when a/mutiple element(s) chemical 
            potential is fixed at certain value. Eg: [("Fe",-1)]
        limits: boundary limits. 
            Eg: [[-10,0],[-10,0]] for a 2 components system
    """

    def __init__(self, entries, elementList, fixed = None, limits = None):
        '''fix is none because we want all mu coordinates'''
        super().__init__(entries, elementList,fixed = None, limits = limits)
        self.ist_entries_dict = self.get_dict_intersection_entries()
    
        self.CH_domain_vertices, self.intersections = self.get_convex_hull_domains(self, entries, elementList, fixed, limits)

    def get_dict_intersection_entries(self):
        '''
        {chemical potential intersection: entries that share this intersection}
        '''
        dict1 = self._stable_domain_vertices
        dict2 = {}
        for i in range(len(self.intersection)+1):
            dict2[i] = []
        for entry,vectors in dict1.items():
            for v in vectors:
                ind = self.intersection.tolist().index(v.tolist())
                dict2[ind].append(entry)
        for entry in self._processed_entries:
            if len(entry.entry.composition.elements) == 1:
                dict2[len(self.intersection)].append(entry)
#         for e in dict2:
#             print(e)
#             print(self.intersection[e])
#             print([iii.name for iii in dict2[e]])
        return dict2
    @staticmethod
    def get_convex_hull_domains(self, CPentries, elementList, fixed = None, limits=None):
        """
            
        Returns:
            Returns a dict of the form {reaction: [boundary_points]}.
        """

            
        C=len(elementList)
        if limits is None:
            limits = []
            for i in range(len(elementList)):
                limits += [[-10, 0]]
        # Get hyperplanes
        hyperplanes = []
        if fixed == None:

            # Create Hyperplanes
            # N = number of entries C =  components number
            # 0 = -G + muAxA + muBxB +muCxC
            # [xB, xC, G] x [muB-muA; muC-muA; -1] = [-muA]
            # FOR HALFSPACEINTERSECTION, the vectors are
            # N x (C+1)  [muB-muA; muC-muA; -1; muA]
            hyperplanes = self.intersection
            mu1 = hyperplanes[:,0].reshape(-1,1)
            mu2 = hyperplanes[:,1].reshape(-1,1)
            mu2Mmu1 = mu2-mu1
            for i in range(2, C):
                muN = hyperplanes[:,i].reshape(-1,1)
                muNMmu1 = muN-mu1
                mu2Mmu1 = np.column_stack((mu2Mmu1,muNMmu1))
            minus1 = np.array([-1 for i in range(len(mu1))]).reshape(-1,1)
            hyperplanes = np.column_stack((mu2Mmu1,minus1,mu1))
#             border_hyperplane = [0 for i in range(C+1)]
#             border_hyperplane[-2] = 1
#             print(border_hyperplane)
#             hyperplanes = np.vstack([hyperplanes, np.array(border_hyperplane)])
            print(hyperplanes)

            internal_point = [1/C for i in range(C)]
            internal_point[-1] = -0.1
            hs_int = HalfspaceIntersection(hyperplanes, np.array(internal_point))
            print("len(self.intersection)", len(self.intersection), self.intersection)
            print("len(hs_int.intersections)", len(hs_int.intersections), hs_int.intersections)
            print("len(hs_int.dual_facets)", len(hs_int.dual_facets), hs_int.dual_facets)
            print()
#             for e in self._stable_domain_vertices:
#                 print(e.name)
#                 print(self._stable_domain_vertices[e])

#             print("self.ist_entries_dict:")
#             for ee in self.ist_entries_dict:
#                 print(ee)
#                 print([eee.name for eee in self.ist_entries_dict[ee]])

            CH_domain_vertices = {entry: [] for entry in CPentries}
            for ist, facet in zip(hs_int.intersections,
                                           hs_int.dual_facets):
                facet_entries = []
                for fa in facet:
                    facet_entries += self.ist_entries_dict[fa]
                re_entry, num = most_frequent(facet_entries)
                print(facet)
                if num == len(facet):
                    CH_domain_vertices[re_entry] = ist
                    print(re_entry.name)
                else:
                    print("None entry")
                
 
            for ee in CH_domain_vertices:
                print(ee.name, CH_domain_vertices[ee])     
            # Remove entries with no intersection
            CH_domain_vertices = {k: v for k, v in CH_domain_vertices.items() if v.all()!=None}
            return CH_domain_vertices, hs_int.intersections
 
        else:
            F = len(fixed)
            fixIndex = [elementList.index(fixed[i][0]) for i in range(F)]
            print("fixIndex", fixIndex)

            for i in range(F):
                print("chemical potential of element",fixed[i][0]," is fixed at",fixed[i][1])
            # Create Hyperplanes
            # We're going to make it a variable length
            # The length of the hyperplane will be C + 1 - 1
            # N = number of entries, C = number of components
            # 0 = G - muAxA - muBxB - muCxC
            # (μB-μA) xB + (μC-μA) xC -G + μA = 0
            # if we fixed xB, then (μB-μA)xB+μA is a constant
            # [xC, G] x [μC-μA; -1] = [-μA-(μB-μA)xB]
            # N x C, [μC-μA; -1; μA+(μB-μA)xB]

            hyperplanes = self.intersection
            print("self.intersection",self.intersection)
            mu1 = hyperplanes[:,0].reshape(-1,1)
#             mu2 = hyperplanes[:,1].reshape(-1,1)
#             mu2Mmu1 = mu2-mu1
            print(0, mu1)
            muNMmu1_list = []
            muFMmu1_list = []
            for i in range(1, C):
                muN = hyperplanes[:,i].reshape(-1,1)
                print(i, muN)
                muNMmu1 = muN-mu1
                if i in fixIndex:
                    muFMmu1_list.append(muNMmu1)
                else:
                    muNMmu1_list.append(muNMmu1)
            mu2Mmu1 = np.column_stack(muNMmu1_list)
            print("mu2Mmu1",mu2Mmu1)
            minus1 = np.array([-1 for i in range(len(mu1))]).reshape(-1,1)
            
            mu1Pmux = mu1.copy()
            for elfix, mum in zip(fixed, muFMmu1_list):
                mu1Pmux = mu1Pmux + mum*elfix[1]
            print("mu1Pmux",mu1Pmux)
            hyperplanes = np.column_stack((mu2Mmu1,minus1,mu1Pmux))
            print(hyperplanes)
            print(elementList)
            remain_comp = 1
            for fe in fixed:
                remain_comp -= fe[1]
            internal_point = [remain_comp/C for i in range(C-F)]
            internal_point[-1] = -0.1
            print(internal_point)
            hs_int = HalfspaceIntersection(hyperplanes, np.array(internal_point))

            print("len(self.intersection)", len(self.intersection), self.intersection)
            print("len(hs_int.intersections)", len(hs_int.intersections), hs_int.intersections.tolist())
            print("len(hs_int.dual_facets)", len(hs_int.dual_facets), hs_int.dual_facets)
            print()
            
            ist_list = hs_int.intersections.tolist()
            real_list = [] 
            print("remain_comp",remain_comp)
            for arr in ist_list:
                print(arr)
                if abs(arr[-1] - (-10)) > 1e-6:
                    print(arr)
                    addornot = True
                    for i in arr[:-1]:
                        if i < 0 or i > remain_comp:
                            if abs(i-remain_comp) > 1e-6 and abs(i) > 1e-6:
                                addornot = False
                    if addornot:
                        print(arr)
                        real_list.append(arr)
            print(np.array(real_list))
            
            complist = []
            for i in real_list:
                complist.append(composition_maker(fixed, elementList, i))
            pdentries = getOrigStableEntriesList(elementList)
            pd = PhaseDiagram(pdentries)
            middle_entries_list = [ComputedEntry(comp,pd.get_hull_energy(comp)) for comp in complist]
            CH_domain_vertices = {}
            reactions_list = []
            for entry in middle_entries_list:
                products = list(pd.get_decomposition(entry.composition).keys())
                reactants = [entry]
                reaction = ComputedReaction(reactants, products)
                reactions_list.append(reaction)

            for re, ist in zip(reactions_list, real_list):
                # key is reaction, reaction.reactants/products is [comp]
                CH_domain_vertices[re] = np.array(ist)
            for e in CH_domain_vertices:
                print(e, CH_domain_vertices[e])

            return CH_domain_vertices, hs_int.intersections

def composition_maker(fixed, elementlist, qhull_data):
    first_ele_comp = 1
    for el in fixed:
        first_ele_comp -= el[1]
    for c in range(len(qhull_data)-1):
        first_ele_comp -= qhull_data[c]
    el_dict = {i[0]:i[1] for i in fixed}
    if first_ele_comp - 1e-6 > 0:
        el_dict[elementlist[0]] = first_ele_comp
    el_list = list(el_dict.keys())
    compstr = ""
    n = 0
    for e in elementlist:
        if e in el_list:
            compstr += e + str(el_dict[e])
        else:
            if elementlist.index(e) != 0:
                if qhull_data[n] - 1e-6 > 0:
                    compstr += e + str(qhull_data[n])
                n += 1
    print(Composition(compstr))
    return Composition(compstr)
            
class SlicePDPlotter(PDPlotter):
    def __init__(
        self,
        phasediagram: PhaseDiagram,
        reactions,
        show_unstable: float = 0.2,
        backend: str = "plotly",
        **plotkwargs,
    ):
        super().__init__(phasediagram,
        show_unstable = show_unstable,
        backend = backend,
        **plotkwargs)
        self.reactions = reactions
    def _create_plotly_stable_labels(self, label_stable=True):
        """
        Creates a (hidable) scatter trace containing labels of stable phases.
        Contains some functionality for creating sensible label positions.

        :return: go.Scatter (or go.Scatter3d) plot
        """
        x, y, z, text, textpositions = [], [], [], [], []
        stable_labels_plot = None
        min_energy_x = None
        offset_2d = 0.005  # extra distance to offset label position for clarity
        offset_3d = 0.01

        energy_offset = -0.1 * self._min_energy
#         for a,e in list(self.pd_plot_data[1].items()):
#             print("haha",e.name)
        if self._dim == 2:
            min_energy_x = min(list(self.pd_plot_data[1].keys()), key=lambda c: c[1])[0]

        for coords, entry in self.pd_plot_data[1].items():
#             if entry.composition.is_element:  # taken care of by other function
#                 continue
            x_coord = coords[0]
            y_coord = coords[1]
            textposition = None

            if self._dim == 2:
                textposition = "bottom left"
                if x_coord >= min_energy_x:
                    textposition = "bottom right"
                    x_coord += offset_2d
                else:
                    x_coord -= offset_2d
                y_coord -= offset_2d
            elif self._dim == 3:
                textposition = "middle center"
                if coords[0] > 0.5:
                    x_coord += offset_3d
                else:
                    x_coord -= offset_3d
                if coords[1] > 0.866 / 2:
                    y_coord -= offset_3d
                else:
                    y_coord += offset_3d

                z.append(self._pd.get_form_energy_per_atom(entry) + energy_offset)

            elif self._dim == 4:
                x_coord = x_coord - offset_3d
                y_coord = y_coord - offset_3d
                textposition = "bottom right"
                z.append(coords[2])

            x.append(x_coord)
            y.append(y_coord)
            textpositions.append(textposition)

#             comp = entry.composition
#             if hasattr(entry, "original_entry"):
#                 comp = entry.original_entry.composition

#             formula = list(comp.reduced_formula)
#             text.append(self._htmlize_formula(formula))
            for re in self.reactions:
                if entry.name == re._reactant_entries[0].name:
                    
                    text.append(re.__str__().split("->")[-1])
        visible = True
        if not label_stable or self._dim == 4:
            visible = "legendonly"

        plot_args = dict(
            text=text,
            textposition=textpositions,
            mode="text",
            name="Labels (stable)",
            hoverinfo="skip",
            opacity=1.0,
            visible=visible,
            showlegend=True,
        )

        if self._dim == 2:
            stable_labels_plot = go.Scatter(x=x, y=y, **plot_args)
        elif self._dim == 3:
            stable_labels_plot = go.Scatter3d(x=y, y=x, z=z, **plot_args)
        elif self._dim == 4:
            stable_labels_plot = go.Scatter3d(x=x, y=y, z=z, **plot_args)

        return stable_labels_plot

    def _create_plotly_element_annotations(self):
        print()
def most_frequent(List): 
    counter = 0
    entry = List[0] 
      
    for i in List: 
        curr_frequency = List.count(i) 
#         print(i.name, curr_frequency)
        if(curr_frequency> counter): 
            counter = curr_frequency 
            entry = i 
#     print("result",entry.name, counter)
    return entry, counter

def generate_entry_label(entry):
    return entry.name
