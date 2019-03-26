Using Pipenv with the Pipfile I included in this directory will get you the proper bayesian_optimization package (For the original version of the /agent scripts) as well as jupyter notebook kernel to work with it. The post-1.0 version of bayesian_optimization requires different code. All to be upgrdaed. This is just to run the original code. Commands to get started:

**pipenv --python 3.7.2**

**pipenv shell**

**pipenv install ipykernel**

**pipenv sync**

**python -m ipykernel install --user --name=my-virtualenv-name**


Replace "my-virtualenv-name" with the name of your pipenv environment. You can see it in the path of the command:

**which python**

Then you can:

**jupyter notebook --ip=127.0.0.1**

Don't forget to switch to the custom agent kernel once the notebook opens.
