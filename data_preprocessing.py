import pandas as pd


def is_label_crf(row, years):
    return 1 if row['EVENT_CRF'] == 1 and row['TIME_CRF'] <= years else 0


def encode_target(df):
    target = pd.DataFrame(columns=['TIME_CRF_2', 'TIME_CRF_5', 'TIME_CRF_10'])

    # apply function to each row
    target['TIME_CRF_2'] = df.apply(lambda row: is_label_crf(row, 2), axis=1)
    target['TIME_CRF_5'] = df.apply(lambda row: is_label_crf(row, 5), axis=1)
    target['TIME_CRF_10'] = df.apply(lambda row: is_label_crf(row, 10), axis=1)

    return target


def fill_missing_values(df):
    # replace missing values with median
    return df.apply(lambda col: col.fillna(value=col.median()), axis=0)


def normalize_columns(df):
    # normalize features between 0 and 1
    return df.apply(lambda col: (col - col.min()) / (col.max() - col.min()), axis=0)


def encode_data(df):
    # drop id column
    df.drop(columns=['IDS'], inplace=True)

    # encode crf for 2,5,10 years
    encoded_target = encode_target(df[['EVENT_CRF', 'TIME_CRF']])
    df.drop(columns=['EVENT_CRF', 'TIME_CRF'], inplace=True)

    # One-hot encode categorical features
    prefix = ['AGE_GROUP', 'SES_GROUP', 'MIGZAR']
    encoded_data = pd.get_dummies(df, prefix=prefix)

    encoded_data = fill_missing_values(encoded_data)
    normalized_df = normalize_columns(encoded_data)

    return normalized_df, encoded_target
