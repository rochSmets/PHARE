
def get_user_inputs(jobname):
    import importlib
    from . import populateDict
    print(jobname)
    jobmodule = importlib.import_module(jobname) #lgtm [py/unused-local-variable]
    populateDict()


