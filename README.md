# LUME LCLS cu inj nn

This repository packages the `LumeLclsCuInjNn` in `lume_lcls_cu_inj_nn/model.py ` for execution with [Prefect](https://docs.prefect.io/) using the flow described in `lume_lcls_cu_inj_nn/flow/flow.py` using the variables:

<!--- The input and output variable tables are replaced when generating the project in template/hooks/post_gen_project.py-->
# input_variables
|        variable name        | type |        default         |
|-----------------------------|------|-----------------------:|
|distgen:r_dist:sigma_xy:value|scalar|       0.413000000000000|
|distgen:t_dist:length:value  |scalar|       7.499772441611215|
|distgen:total_charge:value   |scalar|     250.000000000000000|
|SOL1:solenoid_field_scale    |scalar|       0.246000000000000|
|CQ01:b1_gradient             |scalar|      -0.007400000000000|
|SQ01:b1_gradient             |scalar|      -0.007400000000000|
|L0A_phase:dtheta0_deg        |scalar|      -8.899700000000000|
|L0A_scale:voltage            |scalar|70000000.000000000000000|
|end_mean_z                   |scalar|       4.614700200000000|


# output_variables
|        variable_name         | type |
|------------------------------|------|
|end_n_particle                |scalar|
|end_mean_gamma                |scalar|
|end_sigma_gamma               |scalar|
|end_mean_x                    |scalar|
|end_mean_y                    |scalar|
|end_norm_emit_x               |scalar|
|end_norm_emit_y               |scalar|
|end_norm_emit_z               |scalar|
|end_sigma_x                   |scalar|
|end_sigma_y                   |scalar|
|end_sigma_z                   |scalar|
|end_mean_px                   |scalar|
|end_mean_py                   |scalar|
|end_mean_pz                   |scalar|
|end_sigma_px                  |scalar|
|end_sigma_py                  |scalar|
|end_sigma_pz                  |scalar|
|end_higher_order_energy_spread|scalar|
|end_cov_x__px                 |scalar|
|end_cov_y__py                 |scalar|
|end_cov_z__pz                 |scalar|
|x:y                           |image |
|out_ymax                      |scalar|
|out_xmax                      |scalar|
|out_ymin                      |scalar|
|out_xmin                      |scalar|



## Installation

This package may be installed using pip:
```
pip install git+https://github.com/jacquelinegarrahan/lume-lcls-cu-inj-nn
```


## Dev

Install dev environment:
```
conda env create -f dev-environment.yml
```

Activate your environment:
```
conda activate lume-lcls-cu-inj-nn-dev
```

Install package:
```
pip install -e .
```

Tests can be executed from the root directory using:
```
pytest .
```

## Collecting live EPICS variables
A script (`epics_queue.py`) has been packaged in the root of this repository that may use the LUME-services tools to continually queue model execution on EPICS PV monitors.

### Local execution w/ lume-services
<br>

The environment for the LUME-services `docker-compose` system should be configured as described in the `LUME-services` [demo](https://slaclab.github.io/lume-services/demo/#11-start-services-with-docker-compose).

Launch the compose application using the development environment packaged with this repository and configured as described above:
```
conda activate lume-lcls-cu-inj-nn-dev
lume-services docker start-services
```

In another window, run the model and deployment registration blocks using the notebook in `examples/Run.ipynb`.
```
conda activate lume-lcls-cu-inj-nn-dev
source examples/demo.env
jupyter notebook examples/Run.ipynb
```

In yet another window, configure the EPICS environment using a port forwarding from prod machines.
```
export EPICS_CA_NAME_SERVER_PORT=24666
ssh -fN -L $EPICS_CA_NAME_SERVER_PORT:<LCLS_PROD_HOST> $SLAC_USERNAME@$SLAC_MACHINE
export EPICS_CA_NAME_SERVERS=localhost:$EPICS_CA_NAME_SERVER_PORT
```
Where `EPICS_CA_NAME_SERVER_PORT` may be any available port, not necessarily 24666. The other variables should be provided by someone with context to SLAC systems.

```
conda activate lume-lcls-cu-inj-nn-dev
source examples/demo.env
python epics_queue.py 1 1
```
Where the first argument `1` corresponds to the registered `model_id` and the second `1` corresponds to the registered `deployment_id`.

Once finished, you'll have to kill the port-forwarding process manually by finding the pid:
```
ps aux | grep ssh
kill <pid of your ssh process>
```

### SLAC development network
Use the dev on prod EPICS configuration:

```
conda activate lume-lcls-cu-inj-nn-dev
python epics_queue.py
```



### Note
This README was automatically generated using the template defined in https://github.com/slaclab/lume-services-model-template with the following configuration:

```json
{
    "author": "Jacqueline Garrahan",
    "email": "jacquelinegarrahan@gmail.com",
    "github_username": "jacquelinegarrahan",
    "github_url": "https://github.com/slaclab/lume-lcls-cu-inj-nn",
    "project_name": "LUME LCLS cu inj nn", 
    "project_slug": "lume-lcls-cu-inj-nn", 
    "package": "lume_lcls_cu_inj_nn",
    "model_class": "LumeLclsCuInjNn"
}
```
