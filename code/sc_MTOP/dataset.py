import numpy as np

class __pannuke:

    def __init__(self):
        self.cell_type = ['T', 'I', 'S']

        self.typeDict2 = {
                'neolabe': 0,
                'neopla': 1,
                'inflame': 2,
                'connect': 3,
                'necros': 4,
                'normal': 5
            }
        
        self.typeDict = {
                0: "nolabe",
                1: "neopla",
                2: "inflam",
                3: "connec",
                4: "necros",
                5: "normal"
                }
        
        
        self.cellType_save = {'T': [1],  # Neopla
                                'I': [2],  # Inflam
                                'S': [3],  # Connec
                                'N': [5]}  # Normal
            
    def get_featype_dict(self, nuc_types):
        num_types = len(self.typeDict2)
        nolabeIDs, neoplaIDs, inflamIDs, connecIDs, necrosIDs, normalIDs  = \
        [np.where(np.array(nuc_types) == i)[0].tolist() for i in range(num_types)] 
        featype_dict = {'T-T': [neoplaIDs, neoplaIDs],
                        'I-I': [inflamIDs, inflamIDs],
                        'S-S': [connecIDs, connecIDs],
                        # 'N-N': [normalIDs, normalIDs],
                        'T-I': [neoplaIDs, inflamIDs],
                        'T-S': [neoplaIDs, connecIDs],
                        # 'T-N': [neoplaIDs, normalIDs],
                        'I-S': [inflamIDs, connecIDs],
                        # 'I-N': [inflamIDs, normalIDs],
                        # 'S-N': [connecIDs, normalIDs]
        }
        return featype_dict, neoplaIDs, inflamIDs, connecIDs


class __orion_GMM_7_class:
    def __init__(self):

        self.cell_type = ['T', 'M', 'L', 'V', 'F']

        self.typeDict2 = {
            'undefined': 0,
            "benign": 1,
            'tumor': 2,
            'macrophage': 3,
            'lymphocyte': 4,
            'vascular': 5,
            'fibroblast': 6
            }
        
        self.typeDict = {v: k for k, v in self.typeDict2.items()}
        
        self.cellType_save = {'T': [2],  # Tumor
                        'M': [3],  # Macrophage
                        'L': [4],  # Lymphocyte
                        'V': [5],  # Vascular
                        'F': [6]   # Fibroblast
                        }  

    def get_featype_dict(self, nuc_types):
        num_types = len(self.typeDict2)
        undefinedIDs, benignIDs, tumIDs, macroIDs, lymphIDs, vascuIDs, fibroIDs = \
        [np.where(np.array(nuc_types) == i)[0].tolist() for i in range(num_types)] 
        featype_dict = {'T-T': [tumIDs, tumIDs],
                    'T-M': [tumIDs, macroIDs],
                    'T-L': [tumIDs, lymphIDs],
                    'T-V': [tumIDs, vascuIDs],
                    'T-F': [tumIDs, fibroIDs],
                    'M-M': [macroIDs, macroIDs],
                    'M-L': [macroIDs, lymphIDs],
                    'M-V': [macroIDs, vascuIDs],
                    'M-F': [macroIDs, fibroIDs],
                    'L-L': [lymphIDs, lymphIDs],
                    'L-V': [lymphIDs, vascuIDs],
                    'L-F': [lymphIDs, fibroIDs],
                    'F-V': [fibroIDs, vascuIDs],
                    'F-F': [fibroIDs, fibroIDs],
                    }
        return featype_dict, tumIDs, lymphIDs, fibroIDs, macroIDs, vascuIDs
    
def get_dataset(name):
    """Return a pre-defined dataset object associated with `name`."""
    name_dict = {
        "pannuke": lambda: __pannuke(),
        "orion_GMM_7_class": lambda: __orion_GMM_7_class(),
    }
    if name in name_dict:
        return name_dict[name]()
    else:
        assert False, "Unknown dataset `%s`" % name



   



