import copy
from typing import Dict
from lume_model.variables import InputVariable, OutputVariable

from lume_lcls_cu_inj_nn import INPUT_VARIABLES, OUTPUT_VARIABLES
from lume_lcls_cu_inj_nn.files import MODEL_FILE
import numpy as np

class LCLSCuInjNN():
        
    def __init__(self):
        """Initialize the model. If additional settings are required, they can be 
        passed and handled here. For models that are wrapping model loads
        from other frameworks, this can be used for loading weights, referencing
        data files, etc.
        
        """
        print('hi')

    # EVALUATE implemented on KerasModel base class
    def format_input(self, input_dictionary):
        scalar_inputs = np.array([
            input_dictionary['distgen:r_dist:sigma_xy:value'],
            input_dictionary['distgen:t_dist:length:value'],
            input_dictionary['distgen:total_charge:value'],
            input_dictionary['SOL1:solenoid_field_scale'],
            input_dictionary['CQ01:b1_gradient'],
            input_dictionary['SQ01:b1_gradient'],
            input_dictionary['L0A_phase:dtheta0_deg'],
            input_dictionary['L0A_scale:voltage'],
            input_dictionary['end_mean_z']
            ]).reshape((1,9))


        model_input = [scalar_inputs]
        return  model_input


    def parse_output(self, model_output):        
        parsed_output = {}
        parsed_output["x:y"] = model_output[0][0].reshape((50,50))

        # NTND array attributes MUST BE FLOAT 64!!!! np.float() should be moved to lume-epics
        parsed_output["out_xmin"] = np.float64(model_output[1][0][0])
        parsed_output["out_xmax"] = np.float64(model_output[1][0][1])
        parsed_output["out_ymin"] = np.float64(model_output[1][0][2])
        parsed_output["out_ymax"] = np.float64(model_output[1][0][3])

        parsed_output.update(dict(zip(self.output_variables.keys(), model_output[2][0].T)))

        return parsed_output
