Test fixtures are generated in-memory by `tests/`, so nothing malicious is
ever committed to this repository. To build one by hand:

    python -c "import pickle,os;open('evil.pkl','wb').write(pickle.dumps(type('E',(),{'__reduce__':lambda s:(os.system,('id',))})()))"

Then audit it -- weightsentry reports the callable without loading it:

    weightsentry evil.pkl
