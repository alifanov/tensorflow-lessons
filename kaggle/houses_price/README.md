# Experiment plan

# Best: 0.1639

## Reduce columns

Action: Check correlation and choose only that high correlate

Result:

## Add columns/features

Action: Square, log of some features

Result:

## Add learning rate scheduler

Action: Use LearninRateScheduler

Result:

## Less layers

Action: Remove layers to Sequence

Result:

## More width of hidden layers

Action: Increase width of hidden layers

Result:

## More layers

Action: Add layers to Sequence

Result:

--------------------------------------------------------------------------

## Small test dataset

Action: Use small test dataset (0.1)

Result: Bad (0.1717)

## Scale Y

Action: Use MinMaxScaler

Result: Very bad (1.6)

## scale X

Action: Use MinMaxScaler

Result: Bad (0.2617)


## Scale Y

Action: Use StandardScaler

Result: Bad (0.43)

## scale X

Action: Use StandardScaler

Result: Bad (0.2156)