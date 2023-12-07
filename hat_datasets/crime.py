import numpy as np
import pandas as pd
from os import path
from urllib import request
from sklearn.preprocessing import MinMaxScaler

from utils import save_generated_data,get_data_info

class CrimeDataset():
    def __init__(self):

        print("load communities")
        data_dir = 'dataset'
        data_file = path.join(data_dir, 'communities.data')
        column_names = [
        'communityname', 'state', 'countyCode', 'communityCode', 'fold', 'population', 'householdsize', 'racepctblack',
        'racePctWhite', 'racePctAsian', 'racePctHisp', 'agePct12t21', 'agePct12t29', 'agePct16t24', 'agePct65up',
        'numbUrban', 'pctUrban', 'medIncome', 'pctWWage', 'pctWFarmSelf', 'pctWInvInc', 'pctWSocSec', 'pctWPubAsst',
        'pctWRetire', 'medFamInc', 'perCapInc', 'whitePerCap', 'blackPerCap', 'indianPerCap', 'AsianPerCap',
        'OtherPerCap', 'HispPerCap', 'NumUnderPov', 'PctPopUnderPov', 'PctLess9thGrade', 'PctNotHSGrad', 'PctBSorMore',
        'PctUnemployed', 'PctEmploy', 'PctEmplManu', 'PctEmplProfServ', 'PctOccupManu', 'PctOccupMgmtProf',
        'MalePctDivorce', 'MalePctNevMarr', 'FemalePctDiv', 'TotalPctDiv', 'PersPerFam', 'PctFam2Par', 'PctKids2Par',
        'PctYoungKids2Par', 'PctTeen2Par', 'PctWorkMomYoungKids', 'PctWorkMom', 'NumKidsBornNeverMar',
        'PctKidsBornNeverMar', 'NumImmig', 'PctImmigRecent', 'PctImmigRec5', 'PctImmigRec8', 'PctImmigRec10',
        'PctRecentImmig', 'PctRecImmig5', 'PctRecImmig8', 'PctRecImmig10', 'PctSpeakEnglOnly', 'PctNotSpeakEnglWell',
        'PctLargHouseFam', 'PctLargHouseOccup', 'PersPerOccupHous', 'PersPerOwnOccHous', 'PersPerRentOccHous',
        'PctPersOwnOccup', 'PctPersDenseHous', 'PctHousLess3BR', 'MedNumBR', 'HousVacant', 'PctHousOccup',
        'PctHousOwnOcc', 'PctVacantBoarded', 'PctVacMore6Mos', 'MedYrHousBuilt', 'PctHousNoPhone', 'PctWOFullPlumb',
        'OwnOccLowQuart', 'OwnOccMedVal', 'OwnOccHiQuart', 'OwnOccQrange', 'RentLowQ', 'RentMedian', 'RentHighQ',
        'RentQrange', 'MedRent', 'MedRentPctHousInc', 'MedOwnCostPctInc', 'MedOwnCostPctIncNoMtg', 'NumInShelters',
        'NumStreet', 'PctForeignBorn', 'PctBornSameState', 'PctSameHouse85', 'PctSameCity85', 'PctSameState85',
        'LemasSwornFT', 'LemasSwFTPerPop', 'LemasSwFTFieldOps', 'LemasSwFTFieldPerPop', 'LemasTotalReq',
        'LemasTotReqPerPop', 'PolicReqPerOffic', 'PolicPerPop', 'RacialMatchCommPol', 'PctPolicWhite', 'PctPolicBlack',
        'PctPolicHisp', 'PctPolicAsian', 'PctPolicMinor', 'OfficAssgnDrugUnits', 'NumKindsDrugsSeiz',
        'PolicAveOTWorked', 'LandArea', 'PopDens', 'PctUsePubTrans', 'PolicCars', 'PolicOperBudg',
        'LemasPctPolicOnPatr', 'LemasGangUnitDeploy', 'LemasPctOfficDrugUn', 'PolicBudgPerPop', 'murders',
        'murdPerPop', 'rapes', 'rapesPerPop', 'robberies', 'robbbPerPop', 'assaults', 'assaultPerPop', 'burglaries',
        'burglPerPop', 'larcenies', 'larcPerPop', 'autoTheft', 'autoTheftPerPop', 'arsons', 'arsonsPerPop',
        'ViolentCrimesPerPop', 'nonViolPerPop'
        ]
        dataset = pd.read_csv(data_file, sep=',', header=None, names=column_names)

        # remove features that are not predictive
        dataset.drop(['communityname', 'countyCode', 'communityCode', 'fold'], axis=1, inplace=True)

        # remove all other potential goal variables
        dataset.drop(
            [
                'murders', 'murdPerPop', 'rapes', 'rapesPerPop', 'robberies', 'robbbPerPop', 'assaults',
                'assaultPerPop', 'burglaries', 'burglPerPop', 'larcenies', 'larcPerPop', 'autoTheft',
                'autoTheftPerPop', 'arsons', 'arsonsPerPop', 'nonViolPerPop'
            ], axis=1, inplace=True
        )

        dataset.replace(to_replace='?', value=np.nan, inplace=True)
        print(dataset.shape)
        print(dataset['ViolentCrimesPerPop'])
        # drop rows with missing labels
        dataset.dropna(axis=0, subset=['ViolentCrimesPerPop'], inplace=True)

        # drop columns with missing values
        dataset.dropna(axis=1, inplace=True)
        print(dataset.shape)


        continuous_vars = []
        self.categorical_columns = []
        for col in dataset.columns:
            if dataset[col].isnull().sum() > 0:
                dataset.drop(col, axis=1, inplace=True)
            else:
                if dataset[col].dtype == np.object:
                    self.categorical_columns += [col]
                else:
                    continuous_vars += [col]
        
        self.c_vars = continuous_vars
        self.continuous_columns = [dataset.columns.get_loc(var) for var in continuous_vars]
        self.d_columns = [dataset.columns.get_loc(var) for var in self.categorical_columns]
        self.data = dataset[self.c_vars + self.categorical_columns]
        
    def transform(self):
        self.columns_name = self.data.columns
        self.output_info = get_data_info(self.data ,self.categorical_columns)
        
        self.data = pd.get_dummies(self.data, columns=self.categorical_columns, prefix_sep='=')
        #onehot:numpy array
        self.scaler = MinMaxScaler()
        self.data[self.c_vars] = self.scaler.fit_transform(self.data[self.c_vars])
        print('Attributes', self.columns_name)
        print('Data info:',self.output_info)
        #change to numpy array
        data_np = self.data.values

        #change range to [-1,1]
        #data_np = (data_np[:,:] - 0.5)*2

        return data_np
    
    def inverse_transform(self,data):
        #data: numpy array
        data_i =[]
        st = 0
        #change range to [0,1]
        #data_c  = data[:,:]/2 + 0.5

        data_c = self.scaler.inverse_transform(data[:,self.continuous_columns])
        for item in self.output_info:
            if item[1] == 'softmax':
                ed = st +item[0]
                data_a = np.argmax(data[:, st:ed], axis =1)
                data_a.resize((len(data),1))
                data_i.append(data_a)
                st = ed
            elif item[1]=='tanh':
                ed = st +item[0] 
                st = ed
            else:
                assert 0
        data_i = np.concatenate(data_i, axis=1).astype('int64')
        data_i = np.concatenate((data_c, data_i), axis=1)
        #save_generated_data(data=data_i,name=save_name,column_name=self.columns_name)
        print("Inverse transform completed!")
        return data_i