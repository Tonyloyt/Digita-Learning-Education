import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder , MinMaxScaler
le = LabelEncoder()
scaler = MinMaxScaler(feature_range=(0, 1))

import warnings
warnings.filterwarnings('ignore')


def preprocess_performance(df, option):
    
    #Drop values based on operational options
    if (option == "Online"):
        df['gender'] = le.fit_transform(df['gender'])
        df['NationalITy'] = le.fit_transform(df['NationalITy'])
        df['PlaceofBirth'] = le.fit_transform(df['PlaceofBirth'])
        df['StageID'] = le.fit_transform(df['StageID'])
        df['GradeID'] = le.fit_transform(df['GradeID'])
        df['SectionID'] = le.fit_transform(df['SectionID'])
        df['Topic'] = le.fit_transform(df['Topic'])
        df['Semester'] = le.fit_transform(df['Semester'])
        df['Relation'] = le.fit_transform(df['Relation'])
        df['ParentAnsweringSurvey'] = le.fit_transform(df['ParentAnsweringSurvey'])
        df['ParentschoolSatisfaction'] = le.fit_transform(df['ParentschoolSatisfaction'])
        df['StudentAbsenceDays'] = le.fit_transform(df['StudentAbsenceDays'])

    elif (option == "Batch"):
        pass
        df = df[['gender', 'NationalITy', 'PlaceofBirth', 'StageID', 'GradeID','SectionID', 'Topic', 'Semester', 'Relation', 'raisedhands','VisITedResources', 'AnnouncementsView', 'Discussion',
       'ParentAnsweringSurvey', 'ParentschoolSatisfaction', 'StudentAbsenceDays',]]
       
        #convertion
        df['gender'] = le.fit_transform(df['gender'])
        df['NationalITy'] = le.fit_transform(df['NationalITy'])
        df['PlaceofBirth'] = le.fit_transform(df['PlaceofBirth'])
        df['StageID'] = le.fit_transform(df['StageID'])
        df['GradeID'] = le.fit_transform(df['GradeID'])
        df['SectionID'] = le.fit_transform(df['SectionID'])
        df['Topic'] = le.fit_transform(df['Topic'])
        df['Semester'] = le.fit_transform(df['Semester'])
        df['Relation'] = le.fit_transform(df['Relation'])
        df['ParentAnsweringSurvey'] = le.fit_transform(df['ParentAnsweringSurvey'])
        df['ParentschoolSatisfaction'] = le.fit_transform(df['ParentschoolSatisfaction'])
        df['StudentAbsenceDays'] = le.fit_transform(df['StudentAbsenceDays'])
    else:        #Encoding the other categorical categoric features with more than two categories

        print("Incorrect operational options")

     #feature scaling
    cols = df.columns
    scaled_df = scaler.fit_transform(df[cols])

    return scaled_df


def preprocess_dropout(df, option):

    #Drop values based on operational options
    if (option == "Online"):
        df['Hour_per_week_wriassignment'] = le.fit_transform(df['Hour_per_week_wriassignment'])
        df['time_to_reach college'] = le.fit_transform(df['time_to_reach college'])
        df['Average attendence'] = le.fit_transform(df['Average attendence'])
        df['Internet at home'] = le.fit_transform(df['Internet at home'])
        df['2 hr straight lecture'] = le.fit_transform(df['2 hr straight lecture'])
        df['submission on time'] = le.fit_transform(df['submission on time'])
        df['Five lecture straight,woulf you attend all?'] = le.fit_transform(df['Five lecture straight,woulf you attend all?'])
        df['Five hr practical staight,do you attend all'] = le.fit_transform(df['Five hr practical staight,do you attend all'])
        df['Feedback of teacher'] = le.fit_transform(df['Feedback of teacher'])
        df['preffered transportatin to college'] = le.fit_transform(df['preffered transportatin to college'])
        df['Enrolled to coaching class'] = le.fit_transform(df['Enrolled to coaching class'])

    elif (option == "Batch"):
        pass
        df = df[['SEM 1 KT', 'SEM 2 SGPA', 'SEM 2 KT', 'SEM 4 SGPA', 'SEM 5 SGPA','SEM 6 SGPA', 'SEM 7 SGPA', 'SEM 8 SGPA', 'Hour_per_week_wriassignment',
       'time_to_reach college', 'Average attendence', 'Internet at home','2 hr straight lecture', 'submission on time',
       'Five lecture straight,woulf you attend all?','Five hr practical staight,do you attend all', 'Feedback of teacher','preffered transportatin to college', 'Enrolled to coaching class']]
       
        #convertion
        df['Hour_per_week_wriassignment'] = le.fit_transform(df['Hour_per_week_wriassignment'])
        df['time_to_reach college'] = le.fit_transform(df['time_to_reach college'])
        df['Average attendence'] = le.fit_transform(df['Average attendence'])
        df['Internet at home'] = le.fit_transform(df['Internet at home'])
        df['2 hr straight lecture'] = le.fit_transform(df['2 hr straight lecture'])
        df['submission on time'] = le.fit_transform(df['submission on time'])
        df['Five lecture straight,woulf you attend all?'] = le.fit_transform(df['Five lecture straight,woulf you attend all?'])
        df['Five hr practical staight,do you attend all'] = le.fit_transform(df['Five hr practical staight,do you attend all'])
        df['Feedback of teacher'] = le.fit_transform(df['Feedback of teacher'])
        df['preffered transportatin to college'] = le.fit_transform(df['preffered transportatin to college'])
        df['Enrolled to coaching class'] = le.fit_transform(df['Enrolled to coaching class'])
    else:        #Encoding the other categorical categoric features with more than two categories

        print("Incorrect operational options")

    return df