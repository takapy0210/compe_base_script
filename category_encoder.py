import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder


def sk_label_encoder_2df(train_df, test_df, cols):
    # 0~割り振られる
    tmp_df = pd.concat([train_df, test_df], axis=0, sort=False).reset_index(drop=True)
    le = LabelEncoder().fit(tmp_df[cols])
    return pd.DataFrame(le.transform(train_df[cols]), columns=[cols]), pd.DataFrame(le.transform(test_df[cols]),
                                                                                    columns=[cols])


def sk_label_encoder_1df(df, cols):
    """カテゴリ変換
    sklearnのLabelEncoderでEncodingを行う

    Args:
        df: カテゴリ変換する対象のデータフレーム
        cols (list of str): カテゴリ変換する対象のカラムリスト

    Returns:
        pd.Dataframe: dfにカテゴリ変換したカラムを追加したデータフレーム
    """
    # 0~割り振られる
    for col in cols:
        le = LabelEncoder()
        not_null = df[col][df[col].notnull()]  # nullのデータは変換対象外
        df[col] = pd.Series(le.fit_transform(not_null), index=not_null.index)
        # df[col + '_sk_lbl'] = pd.Series(le.fit_transform(not_null), index=not_null.index)
    return df


def target_encoder_mean(train_df, test_df, cols, target):
    target_mean = train_df.groupby(cols)[target].mean()
    return pd.DataFrame(train_df[cols].map(target_mean)), pd.DataFrame(test_df[cols].map(target_mean))


def target_encoder_std(train_df, test_df, cols, target):
    target_mean = train_df.groupby(cols)[target].std()
    return pd.DataFrame(train_df[cols].map(target_mean)), pd.DataFrame(test_df[cols].map(target_mean))


def ce_ordinal_encoder_tt(train_df, test_df, cols):
    # 1~割り振られる
    tmp_df = pd.concat([train_df, test_df], axis=0, sort=False).reset_index(drop=True)
    ce_oe = ce.OrdinalEncoder(cols=cols, handle_unknown='impute')
    tmp_df = ce_oe.fit_transform(tmp_df[cols])
    tmp_train = tmp_df.iloc[:len(train_df), :]
    tmp_test = tmp_df.iloc[len(train_df):, :]
    return tmp_train, tmp_test


def ce_ordinal_encoder(df, cols):
    # 1~割り振られる
    tmp_df = df.copy()
    ce_oe = ce.OrdinalEncoder(cols=cols, handle_unknown='impute')
    tmp_df[cols] = ce_oe.fit_transform(tmp_df[cols])
    return tmp_df


def ce_one_hot_encoder(train_df, test_df, cols):
    tmp_df = pd.concat([train_df, test_df], axis=0, sort=False).reset_index(drop=True)
    ce_ohe = ce.OneHotEncoder(cols=cols, handle_unknown='impute')
    tmp_df = ce_ohe.fit_transform(tmp_df[cols])
    tmp_train = tmp_df.iloc[:len(train_df), :]
    tmp_test = tmp_df.iloc[len(train_df):, :]
    return tmp_train, tmp_test


def ce_binary_encoder(train_df, test_df, cols):
    tmp_df = pd.concat([train_df, test_df], axis=0, sort=False).reset_index(drop=True)
    ce_bine = ce.BinaryEncoder(cols=cols, handle_unknown='impute')
    tmp_df = ce_bine.fit_transform(tmp_df[cols])
    tmp_train = tmp_df.iloc[:len(train_df), :]
    tmp_test = tmp_df.iloc[len(train_df):, :]
    return tmp_train, tmp_test


def ce_leave_one_out_encoder_v1(train_df, test_df, cols, target):
    # こちらは正真正銘looエンコードする
    ce_loo = ce.LeaveOneOutEncoder(cols=cols)
    tmp_train = ce_loo.fit_transform(X=train_df[cols], y=train_df[target])
    if test_df is not None:
        tmp_test = ce_loo.transform(test_df[cols])
        return tmp_train, tmp_test
    return tmp_train, None


def ce_leave_one_out_encoder_v2(train_df, test_df, cols, target):
    # こちらは自分も含めてtargetの平均でエンコードする
    ce_loo = ce.LeaveOneOutEncoder(cols=cols).fit(X=train_df[cols], y=train_df[target])
    tmp_train = ce_loo.transform(train_df[cols])
    if test_df is not None:
        tmp_test = ce_loo.transform(test_df[cols])
        return tmp_train, tmp_test
    return tmp_train, None


def ce_target_encoder(train_df, test_df, cols, target):
    ce_tgt = ce.TargetEncoder(cols=cols)
    tmp_train = ce_tgt.fit_transform(X=train_df[cols], y=train_df[target])
    if test_df is not None:
        tmp_test = ce_tgt.transform(test_df[cols])
        return tmp_train, tmp_test
    return tmp_train, None


def ce_catboost_encoder(train_df, test_df, cols, target):
    ce_cbe = ce.CatBoostEncoder(cols=cols, random_state=42)
    tmp_train = ce_cbe.fit_transform(X=train_df[cols], y=train_df[target])
    if test_df is not None:
        tmp_test = ce_cbe.transform(test_df[cols])
        return tmp_train, tmp_test
    return tmp_train, None


def ce_jamesstein_encoder(train_df, test_df, cols, target):
    ce_jse = ce.JamesSteinEncoder(cols=cols, drop_invariant=True)
    tmp_train = ce_jse.fit_transform(X=train_df[cols], y=train_df[target])
    if test_df is not None:
        tmp_test = ce_jse.transform(test_df[cols])
        return tmp_train, tmp_test
    return tmp_train, None
