import pandas as pd
import pyreadstat

df_iaq, meta = pyreadstat.read_sas7bdat('./data/HN_IAQ.sas7bdat')

result = pd.concat(
    [
        pyreadstat.read_sas7bdat('./data/HN21_ALL.sas7bdat')[0],
        pyreadstat.read_sas7bdat('./data/HN19_ALL.sas7bdat')[0],
        pyreadstat.read_sas7bdat('./data/hn15_all.sas7bdat')[0],
        pyreadstat.read_sas7bdat('./data/hn14_all.sas7bdat', encoding="latin1")[0],
        pyreadstat.read_sas7bdat('./data/hn13_all.sas7bdat', encoding="latin1")[0],
        pyreadstat.read_sas7bdat('./data/hn12_all.sas7bdat', encoding="latin1")[0],
        pyreadstat.read_sas7bdat('./data/hn11_all.sas7bdat', encoding="latin1")[0],
    ]
)

result = pd.concat(
    [
        pyreadstat.read_sas7bdat('./data/HN21_ALL.sas7bdat')[0],
        pyreadstat.read_sas7bdat('./data/HN20_ALL.sas7bdat')[0],
        pyreadstat.read_sas7bdat('./data/HN19_ALL.sas7bdat')[0],
        pyreadstat.read_sas7bdat('./data/HN18_ALL.sas7bdat')[0],
        pyreadstat.read_sas7bdat('./data/hn17_all.sas7bdat')[0],
        pyreadstat.read_sas7bdat('./data/hn16_all.sas7bdat', encoding="latin1")[0],
        pyreadstat.read_sas7bdat('./data/hn15_all.sas7bdat')[0],
        pyreadstat.read_sas7bdat('./data/hn14_all.sas7bdat', encoding="latin1")[0],
        pyreadstat.read_sas7bdat('./data/hn13_all.sas7bdat', encoding="latin1")[0],
        pyreadstat.read_sas7bdat('./data/hn12_all.sas7bdat', encoding="latin1")[0],
        pyreadstat.read_sas7bdat('./data/hn11_all.sas7bdat', encoding="latin1")[0],
    ]
)

#
result['ID_trimmed'] = result['ID'].str[4:]
result['ID_fam_trimmed'] = result['ID_fam'].str[4:]
df_iaq['ID_trimmed'] = df_iaq['ID'].str[4:]
df_iaq['ID_fam_trimmed'] = df_iaq['ID_fam'].str[4:]

df_21_iaq = pd.merge(result, df_iaq, on=['ID_trimmed', 'ID_fam_trimmed'], how='inner', suffixes=('', '_drop'))
df_21_iaq.drop([col for col in df_21_iaq.columns if col.endswith('_drop')], axis=1, inplace=True)
df_21_iaq.drop_duplicates(inplace=True)

aa = df_iaq[df_iaq["ID_trimmed"] == "215301"]
# => ID, ID_fam 이 같더라도 다른 사람인 것을 확인할 수 있음. 즉 조인 불가.
