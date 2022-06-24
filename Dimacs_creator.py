import os
from typing import List
from xml.etree import ElementTree as ET


class DimacsCreator():
    def __init__(self,sys_name):
        self.sys_name = sys_name

    def _parse_excluded_options(self):
        fm_name = 'FeatureModel.xml'
        sys_dir = '/application/Distance-Based_Data/SupplementaryWebsite/MeasuredPerformanceValues'

        XML_file_path = os.path.join(sys_dir,self.sys_name,fm_name)
        tree = ET.parse(XML_file_path)
        root = tree.getroot()  
        excludes = {}

        for e1 in root.iter('configurationOption'):
            n1 = e1.find('name').text
            ex_list = []
            for n2 in e1.find('excludedOptions').iter('options'):
                ex_list += [n2.text]
            excludes[n1] = ex_list
        return excludes

    def _parse_positionmap(self):
        fm_name = 'FeatureModel.xml'
        sys_dir = '/application/Distance-Based_Data/SupplementaryWebsite/MeasuredPerformanceValues'

        XML_file_path = os.path.join(sys_dir,self.sys_name,fm_name)
        tree = ET.parse(XML_file_path)
        root = tree.getroot()
        
        #if "root" in [e.find('name').text for e in root.iter('configurationOption')]:
        ret = {e.find('name').text :i+1 for i,e in enumerate(root.iter('configurationOption'))}
        #else:
        #    ret = {"root":1}
        #    ret.update({e.find('name').text :i+2 for i,e in enumerate(root.iter('configurationOption'))})

        return ret

    def _parse_non_optional_features(self):
        fm_name = 'FeatureModel.xml'
        sys_dir = '/application/Distance-Based_Data/SupplementaryWebsite/MeasuredPerformanceValues'

        XML_file_path = os.path.join(sys_dir,self.sys_name,fm_name)
        tree = ET.parse(XML_file_path)
        root = tree.getroot()
        non_optionals = []
        for e in root.iter('configurationOption'):
            n = e.find('name').text
            b = e.find('optional').text
            if b=="False":
                non_optionals+=[n]
        #if not "root" in non_optionals:
        #    non_optionals = ["root"]+non_optionals
        return non_optionals

    def _parse_implied_options(self):
        fm_name = 'FeatureModel.xml'
        sys_dir = '/application/Distance-Based_Data/SupplementaryWebsite/MeasuredPerformanceValues'

        XML_file_path = os.path.join(sys_dir,self.sys_name,fm_name)
        tree = ET.parse(XML_file_path)
        root = tree.getroot()  
        impliedOptions = {}

        for e1 in root.iter('configurationOption'):
            n1 = e1.find('name').text
            im_list = []
            for n2 in e1.find('impliedOptions').iter('options'):
                im_list += [n2.text]
            impliedOptions[n1] = im_list
        return impliedOptions

    def _parse_parents(self):
        fm_name = 'FeatureModel.xml'
        sys_dir = '/application/Distance-Based_Data/SupplementaryWebsite/MeasuredPerformanceValues'

        XML_file_path = os.path.join(sys_dir,self.sys_name,fm_name)
        tree = ET.parse(XML_file_path)
        root = tree.getroot()  
        parents = {}

        for e1 in root.iter('configurationOption'):
            n = e1.find('name').text
            p = e1.find('parent').text
            if p:
                if p.strip() != "":
                    parents[n] = p.strip()
        return parents

    def _parse_children(self):
        children = {}
        for child,parent in self._parse_parents().items():
            if(parent in children):
                children[parent] += [child]
            else:
                children[parent] = [child]
        return children

    def create(self):
        # 1 based positionmap
        dm_position_map = self._parse_positionmap()
        excludes = self._parse_excluded_options()
        implicationen = self._parse_implied_options()
        non_optionals = self._parse_non_optional_features()
        children = self._parse_children()
        
        parents = self._parse_parents()
        restrictions = []

        for non_optional in non_optionals:
            if non_optional in excludes:
                if not excludes[non_optional] == []:
                    continue
            restrictions+=[str(dm_position_map[non_optional])]

        for option1 in excludes:
            for option2 in excludes[option1]:
                #if dm_position_map[option1]>dm_position_map[option2]:
                    #continue
                restrictions+=["-" + str(dm_position_map[option1]) + " -" + str(dm_position_map[option2])]

        for option1 in implicationen:
            for option2 in implicationen[option1]:
                restrictions+=["-" + str(dm_position_map[option1]) + " " + str(dm_position_map[option2])]

        for p in children:
            restrictions+=[" ".join(
                    [str(dm_position_map[c])
                    for c in children[p]
                    ]
                )+" -"+str(dm_position_map[p])
            ]
        
        for key, value in parents.items():
            restrictions+=[str(dm_position_map[value]) + " -" + str(dm_position_map[key])]

        lines =[]
        for key, value in dm_position_map.items():
            lines +=["c "+str(value)+" "+key]
        lines += ["p cnf "+ str(len(dm_position_map)) +" "+str(len(restrictions))]
        for res in restrictions:
            lines+= [res+" 0"]
        
        f = open(os.path.join("dimacs",self.sys_name+".dimacs"), "w")
        f.write("\n".join(lines))
        f.close()