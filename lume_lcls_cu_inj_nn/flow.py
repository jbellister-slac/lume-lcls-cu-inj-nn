from typing import Dict

from prefect import flow, get_run_logger, task

from lume_services.results import Result
from lume_services.tasks import (
    configure_lume_services,
    prepare_lume_model_variables,
    check_local_execution,
    SaveDBResult,
    LoadDBResult,
    LoadFile,
    SaveFile,
)
from lume_services.files import TextFile
from lume_model.variables import InputVariable, OutputVariable

from lume_lcls_cu_inj_nn.model import LCLSCuInjNN
from lume_lcls_cu_inj_nn import INPUT_VARIABLES, CU_INJ_MAPPING_TABLE

import torch
import matplotlib.pyplot as plt

from lume_model.utils import variables_from_yaml
from lume_model.models.torch_model import TorchModel
from lume_model.models.torch_module import TorchModule
import os


@task()
def format_result(
    input_variables: Dict[str, InputVariable],
    output_variables,
    output_variables_names
):
    outputs = {}
    logger = get_run_logger()
    output_variables = output_variables.tolist()
    output_variables_names = list(output_variables_names.keys())
    logger.info('Output Variables List - ', output_variables_names)

    for i in range(len(output_variables)):
        outputs[output_variables_names[i]] = output_variables[i]

    return Result(inputs=input_variables, outputs=outputs)


@task()
def evaluate(formatted_input_vars, lume_model):
    logger = get_run_logger()
    predictions = lume_model.evaluate(formatted_input_vars)
    logger.info(f'Predictions - {predictions}')
    return predictions


#TODO: Save result
#save_db_result_task = SaveDBResult()


@task()
def load_input(var_name, parameter):
    # Confirm Inputs are Correctly Loaded!
    logger = get_run_logger()
    if parameter.value is None:
        parameter.value = parameter.default
    logger.info(f'Loaded {var_name} with value {parameter}')
    return parameter


@flow(name="lume-lcls-cu-inj-nn")
def lume_lcls_cu_inj_nn_flow():
    print('Starting Flow Run')
    # CONFIGURE LUME-SERVICES
    # see https://slaclab.github.io/lume-services/workflows/#configuring-flows-for-use-with-lume-services
    
    #configure = configure_lume_services()

    # CHECK WHETHER THE FLOW IS RUNNING LOCALLY
    # If the flow runs using a local backend, the results service will not be available
    #running_local = check_local_execution()
    #running_local.set_upstream(configure)

    input_variable_parameter_dict = {}
    
    for var in INPUT_VARIABLES:
        input_variable_parameter_dict[var.name] = load_input(var.name, var)

    print('Input Variable Parameters - ', input_variable_parameter_dict)
    
    if os.path.exists('model/'):
        TORCH_MODEL_PATH = 'model/'
    elif os.path.exists('/lume-lcls-cu-inj-nn/lume_lcls_cu_inj_nn/model/'):
        #This is the Docker Path
        TORCH_MODEL_PATH = '/lume-lcls-cu-inj-nn/lume_lcls_cu_inj_nn/model/'
    elif os.path.exists('lume_lcls_cu_inj_nn/model/'):
        TORCH_MODEL_PATH = 'lume_lcls_cu_inj_nn/model/'
    else:
        print('Path Not Found')

    print('Reached Here with TORCH MODEL PATH - ', TORCH_MODEL_PATH)
    print(os.listdir())
  
    saved_torch_model = torch.load(TORCH_MODEL_PATH+"model.pt")  
    input_transformer = torch.load(TORCH_MODEL_PATH+"input_transformer.pt")
    output_transformer = torch.load(TORCH_MODEL_PATH+"output_transformer.pt")
    input_variables_names, output_variables_names = variables_from_yaml(TORCH_MODEL_PATH+"variables.yml")

    # create lume model
  
    lume_model = TorchModel(
        model=saved_torch_model,
        input_variables=input_variables_names,
        output_variables=output_variables_names,
        input_transformers=[input_transformer],
        output_transformers=[output_transformer],
    )
    lume_module = TorchModule(
        model=lume_model
    )

    output_variables = evaluate(input_variable_parameter_dict, lume_model)

    print(f'Result: {output_variables}')

    return output_variables

    # SAVE RESULTS TO RESULTS DATABASE, requires LUME-services results backend 
    #if not running_local:
    #    # CREATE LUME-services Result object
    #    formatted_result = format_result(
    #        input_variables=input_variable_parameter_dict, output_variables=output_variables, output_variables_names=output_variables_names
    #    )

        # RUN DATABASE_SAVE_TASK
    #    saved_model_rep = save_db_result_task(formatted_result)
    #    saved_model_rep.set_upstream(configure)


def get_flow():
    return flow

if __name__ == '__main__':
    lume_lcls_cu_inj_nn_flow.serve(name="lume-nn-test")
