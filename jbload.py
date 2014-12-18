from pandas import DataFrame, concat
from numpy import unique
import jbprep
import pdb

def myopic_exp0(db_url, criterion):
    df = jbprep.sql2pandas(db_url, 'myopic_exp0', criterion)
    df['experiment'] = 'myopic_exp0'

    # order properly
    SORTORDER = ['condition','counterbalance','workerid', 'round']
    df = df.sort(SORTORDER)

    # fields saved as lists, but actually only containing scalars
    LO1NUMBER2SCALARFIELDS = {'samX', 'samY'}
    df = jbprep.lo1number2scalar(df, LO1NUMBER2SCALARFIELDS)

    # make fields with lists of numbers numpy arrays
    LONUMBERFIELDS = {'d2locsX', 'd2locsY', 'pxObs', 'pyObs'}
    df = jbprep.lonumbers2nparray(df, LONUMBERFIELDS)

    # rename columns
    OLDNEWCOLNAMES = {'xObs': 'obsX',
                    'yObs': 'obsY',
                    'pxObs': 'pobsX',
                    'pyObs': 'pobsY',
                    'LENSCALE': 'lenscale',
                    'NOISEVAR2': 'noisevar2',
                    'RNGSEED': 'rngseed',
                    'SIGVAR': 'sigvar'}
    df.rename(columns=OLDNEWCOLNAMES, inplace=True)

    # convert any improperly converted fields
    FIELDTYPES = {'lenscale': 'float64',
                'noisevar2': 'float64',
                'rngseed': 'int64',
                'sigvar': 'float64',
                'condition': 'int64',
                'counterbalance': 'int64',
                'd2locsX': 'O',
                'd2locsY': 'O',
                'pobsX': 'O',
                'pobsY': 'O',
                'obsX': 'O',
                'obsY': 'O',
                'drillX': 'float64',
                'drillY': 'float64',
                'psamX': 'float64',
                'psamY': 'float64',
                'samX': 'float64',
                'samY': 'float64',
                'expScore': 'float64',
                'nObs': 'int64',
                'round': 'int64',
                'roundGross': 'float64',
                'roundNet': 'float64',
                'workerid': 'O'}
    df = jbprep.enforceFieldTypes(df, FIELDTYPES)


    # convert condition to a factor
    df.condition = df.condition.astype(str)

    # something got goofed and psiTurk save pxObs and pyObs but not xObs and yObs.
    # Have to hack a fix here instead
    SCREENW = 1028
    SCREENH = 784
    GROUNDLINEY = SCREENH - SCREENH*0.9
    YMIN = -3
    YMAX = 3
    df['obsX'] = df['pobsX'].apply(lambda row: jbprep.pix2mathX(row, SCREENW))
    df['obsY'] = df['pobsY'].apply(lambda row:
            jbprep.pix2mathY(row, SCREENH, GROUNDLINEY, YMIN, YMAX))

    # remove extra info
    # (1 drops along cols, 0 along rows)
    DROPCOLS = ['uniqueid', 'hitid', 'assignmentid', 'pdrillX', 'pdrillY',
                'pobsX', 'pobsY', 'psamX', 'psamY']
    df = df.drop(DROPCOLS, 1)

    return df


def noChoice_exp0(db_url, criterion):
    ################################
    #  PREPROCESSING DATA
    ################################
    df = jbprep.sql2pandas(db_url, 'noChoice_exp0', criterion)
    df['experiment'] = 'noChoice_exp0'

    # order properly
    SORTORDER = ['condition','counterbalance','workerid', 'round']
    df = df.sort(SORTORDER)

    # make fields with lists of numbers numpy arrays
    LONUMBERFIELDS = {'d2locsX', 'd2locsY', 'pxObs', 'pyObs'}
    df = jbprep.lonumbers2nparray(df, LONUMBERFIELDS)

    # rename columns
    OLDNEWCOLNAMES = {'xObs': 'obsX',
                    'yObs': 'obsY',
                    'pxObs': 'pobsX',
                    'pyObs': 'pobsY',
                    'LENSCALE': 'lenscale',
                    'NOISEVAR2': 'noisevar2',
                    'RNGSEED': 'rngseed',
                    'SIGVAR': 'sigvar'}
    df.rename(columns=OLDNEWCOLNAMES, inplace=True)



    # convert condition to a factor
    df.condition = df.condition.astype(str)

    # something got goofed and psiTurk save pxObs and pyObs but not xObs and yObs.
    # Have to hack a fix here instead
    SCREENW = 1028
    SCREENH = 784
    GROUNDLINEY = SCREENH - SCREENH*0.9
    YMIN = -3
    YMAX = 3
    df['obsX'] = df['pobsX'].apply(lambda row: jbprep.pix2mathX(row, SCREENW))
    df['obsY'] = df['pobsY'].apply(lambda row:
            jbprep.pix2mathY(row, SCREENH, GROUNDLINEY, YMIN, YMAX))

    # convert any improperly converted fields
    FIELDTYPES = {'lenscale': 'float64',
                'noisevar2': 'float64',
                'rngseed': 'int64',
                'sigvar': 'float64',
                'condition': 'int64',
                'counterbalance': 'int64',
                'd2locsX': 'O',
                'd2locsY': 'O',
                'pobsX': 'O',
                'pobsY': 'O',
                'obsX': 'O',
                'obsY': 'O',
                'drillX': 'float64',
                'drillY': 'float64',
                'expScore': 'float64',
                'nObs': 'int64',
                'round': 'int64',
                'roundGross': 'float64',
                'roundNet': 'float64',
                'workerid': 'O'}
    df = jbprep.enforceFieldTypes(df, FIELDTYPES)

    # remove extra info
    # (1 drops along cols, 0 along rows)
    DROPCOLS = ['uniqueid', 'hitid', 'assignmentid', 'pdrillX', 'pdrillY',
                'pobsX', 'pobsY']
    df = df.drop(DROPCOLS, 1)

    return df


def chooseSam_noNorm(db_url, criterion):
    ################################
    #  PREPROCESSING DATA
    ################################
    df = jbprep.sql2pandas(db_url, 'chooseSam_noNorm', criterion)
    df['experiment'] = 'chooseSam_noNorm'

    # order properly
    SORTORDER = ['condition','counterbalance','workerid', 'round']
    df = df.sort(SORTORDER)

    # need to change to new format, where each trial is a row
    # (currently, each click is a row)
    # all fields not here will be collapsed into one value for the trial
    LONUMBERFIELDS = ['drillPX', 'drillPY', 'drillX', 'drillY', 'expScore',
                      'sampleInRound']
    OLDNEWSAMSECNAMEPAIRS = {'drillPX': 'psamX',
                   'drillPY': 'psamY',
                   'drillX': 'samX',
                   'drillY': 'samY',
                   'expScore': 'samExpScore'}

    def trialToOneRow(dfs, lonumberfields, oldnewsamsecnamepairs):
        collapsedFields = {field: [dfs[field].values]
                        for field in lonumberfields}
        collapsedTrial = DataFrame(collapsedFields)
        collapsedTrial.rename(columns=oldnewsamsecnamepairs, inplace=True)

        return collapsedTrial

    minidfs = []
    for workerid in unique(df.workerid):
        dfw = df[df.workerid==workerid]
        for trial in unique(dfw.round):
            dft = dfw[dfw.round==trial]
            dfs = dft[dft.roundSection=='sampling']
            dfd = dft[dft.roundSection=='drilling']
            dfsCollapsed = trialToOneRow(dfs, LONUMBERFIELDS, OLDNEWSAMSECNAMEPAIRS)
            dftCollapsed = concat([dfd.reset_index(drop=True), dfsCollapsed], axis=1)
            minidfs.append(dftCollapsed)
    df = concat(minidfs).reset_index(drop=True)

    # make fields with lists of numbers numpy arrays
    # rename columns
    OLDNEWCOLNAMES = {'drillPX': 'pdrillX',
                    'drillPY': 'pdrillY'}
    df.rename(columns=OLDNEWCOLNAMES, inplace=True)


    # convert condition to a factor
    df.condition = df.condition.astype(str)
    df.counterbalance = df.counterbalance.astype(str)

    # convert any improperly converted fields
    FIELDTYPES = {'gpsd': 'float64',
                'roundRandSeed': 'int64',
                'condition': 'str',
                'counterbalance': 'str',
                'drillX': 'float64',
                'drillY': 'float64',
                'samX': 'O',
                'samY': 'O',
                'expScore': 'float64',
                'samExpScore': 'O',
                'round': 'int64',
                # 'roundGross': 'float64',
                # 'roundNet': 'float64',
                'workerid': 'O'}
    df = jbprep.enforceFieldTypes(df, FIELDTYPES)

    # remove extra info
    # (1 drops along cols, 0 along rows)
    DROPCOLS = ['uniqueid', 'hitid', 'assignmentid', 'pdrillX', 'pdrillY',
                'pobsX', 'pobsY']
    df = df.drop(DROPCOLS, 1)

    return df
