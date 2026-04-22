# PED-based scoring function 

PED can be applied for molecular generation's reward design.

Take [REINVENT 3.2](https://github.com/MolecularAI/reinvent-models) as an example, we can implement PED and itegrate it to [reinvent-scoring](https://github.com/MolecularAI/reinvent-scoring):

1. Create a new folder in `reinvent_scoring/scoring/score_components`

2. Put the provided PED scripts there

3. Also include the whole GeoDiff repo in the new folder

4. Edit other scripts' module import to incorporate PED scoring function