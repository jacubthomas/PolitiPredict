Politipredict is in transition from Java to Python; this is just beginning.

The overall algorithm is being built in pieces. To run them, simply clone navigate to reddit and run:

    `python3 main.py` this calls to modules. The first ingests the dataset, preprocesses, partitions,
    trains multiple classifiers on it and pickles work. The latter, loads pickled work and brute force tests the classifiers against the testing set, outputs their accuracy and update mysql db on incorrectly predicted tests.
    
    OLD --------------------------------------------------------------------------------------------
    `python3 naivebayes.py` this attempts to evaluate for bias using trivial word-weighting.

    Verify all lines below line 9 in posORneg.py is commented out. run `python3 posORneg.py`.
    This preprocesses all the data, trains the classifiers involved and saves everything with
    pickles. Then comment out line 9, uncomment code below and run `python3 posORneg.py`. This
    is the first stage of evaluating bias using sentiment - determines if a statement about a 
    topic pos or negative?
    END OLD ----------------------------------------------------------------------------------------
For more info visit: https://quip.com/TbbtAQpmo25S/Design-Doc-PolitiPredict
