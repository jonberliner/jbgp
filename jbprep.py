import numpy as np
import pdb

def sql2pandas(db_url, table_name, locriterion=None):
    """connects to database at db_url and converts psiturk datatable table_name
       to a pandas df.  Only includes trials that meet all criterion functions
       given in locriterion (default takes all trials)"""
    from sqlalchemy import MetaData, Table, create_engine
    from json import loads
    from pandas import DataFrame, concat

    data_column_name = 'datastring'
    # boilerplace sqlalchemy setup
    engine = create_engine(db_url)
    metadata = MetaData()
    metadata.bind = engine
    table = Table(table_name, metadata, autoload=True)
    # make a query and loop through
    s = table.select()
    sql_expData = s.execute()  # all data from the experiment in sqlalchemy format

    # convert sql rows to lodicts, each containing a subject's full experiment
    # fields from orig datatable that you want attached to every trial
    fullExpFields = ['uniqueid', 'assignmentid', 'workerid', 'hitid', 'status', 'condition', 'counterbalance']
    expData = []
    # sql_subData is a subject's full experiment data in sqlalchemy format
    for sql_subData in sql_expData:
        # try:
        if sql_subData:
            try:
                subData = loads(sql_subData[data_column_name])
            except:
                continue
            for field in fullExpFields:
                if field=='condition':
                    subData['condition'] = sql_subData['cond']
                else:
                    subData[field] = sql_subData[field]
            expData.append(subData)
            # except:
            #     print 'BAD'
            #     continue

    # turn from nested list to flat list of trials
    # pdb.set_trace()
    minidicts = []
    for subData in expData:
        for trial in subData['data']:
            #FIXME: at this point, I have no condition!!
            trialdata = trial['trialdata']
            for field in fullExpFields:
                trialdata[field] = subData[field]

            # check if trial valid if any criterion were passed
            includeThisTrial = True
            if locriterion:
                includeThisTrial = meetsCriterion(trialdata, locriterion)

            if includeThisTrial:
                # pdb.set_trace()
                minidicts.append(trialdata)

    # convert minidicts into dataframe!
    df = DataFrame(minidicts)
    # get rid of residue from minidfs
    df.reset_index(drop=True, inplace=True)
    return df


def meetsCriterion(obj, locrit):
    """passes an object to a list of criterion check functions and returns true
       if all checks return true"""
    for crit in locrit:
        if not crit(obj):
            return False
    return True


def pix2mathX(x, screenW):
    """converts from game pixels to math used to generate along x-axis"""
    return x / screenW


def pix2mathY(y, screenH, groundLineY, ymin, ymax):
    """converts from game pixels to math used to generate along y-axis"""
    groundline2bottom = screenH - groundLineY
    yrange = np.ptp([ymin, ymax])

    y -= groundLineY
    y /= groundline2bottom
    y *= yrange
    y += ymin

    return y


def lonumbers2nparray(df, lonumberFields, ftype=None):
    for f in lonumberFields:
        df[f] = df[f].apply(lambda l: np.array(l))

    ## TODO: modularized this out.  verify that this is cool
    # if type(ftype) is type:  # expand to hash w same ftype for each field
    #     ftype = {f: ftype for f in lonumberFields}
    # if ftype:  # convert fields if hash is none
    #     df = convertNumpyFields(df, ftype)
    return df


def convertNumpyFields(df, doFieldTypePairs):
    for f, ftype in doFieldTypePairs:
        df[f] = df[f].apply(lambda npa: npa.astype(ftype))
    return df


def lo1number2scalar(df, lo1numberFields, ftype=None):
    """takes fields that are stored as lists, but actually only contain
    a scalar, extracts from lists, leaving only the scalar, also can
    optionally to field type conversion (default no conversion)"""
    for f in lo1numberFields:
        df[f] = df[f].apply(lambda l: np.array(l[0]))
    return df


def enforceFieldTypes(df, fieldTypes):
    """verifies that cols of df that match keys in fieldTypes match the
    matching values in fieldTypes.  Converts them if they don't match"""
    for key, ftype in fieldTypes.iteritems():
        if str(df[key].dtype) is not ftype:
            # TODO: verify that we don't need this explicit None replacement
            # df[key] = df[key].replace('None', np.nan)
            # try:
            df[key] = df[key].astype(ftype)
            # except:
            #     pdb.set_trace()

    return df

