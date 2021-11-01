#Import libraries
import streamlit as st
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from PIL import Image


from sklearn.preprocessing import LabelEncoder,MinMaxScaler 
le = LabelEncoder()
scaler = MinMaxScaler(feature_range=(0, 1))

#Saving best model 
import joblib

import warnings
warnings.filterwarnings('ignore')


#load the model from models dir

model = joblib.load(r"models/student_performance/model_sp.sav")
model2 = joblib.load(r"models/student_dropout/model_sd.sav")

#Import python scripts
from preprocessing import preprocess_performance,preprocess_dropout


def main():
    #Setting Application title
    st.title('Digital Learning & Education App')

      #Setting Application description
    st.markdown("""
     :dart:  This Streamlit app is made to predict the performance of a students and likehood of a student to dropout from school due to diffrent factor
    The application is functional for both online prediction and batch data prediction. \n
    """)
    st.markdown("<h3></h3>", unsafe_allow_html=True)

     #Setting Application sidebar default
    image = Image.open('DLE1.jpeg')
    page = st.sidebar.selectbox("Choose performance prediction  or  dropout prediction", ['Performance Prediction', 'Dropout Prediction'])
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?", ("Online", "Batch"))
    st.sidebar.image(image)

    if page == 'Performance Prediction':
        st.title('Student Performance Prediction')
        st.text('Select a page in the sidebar')


        if add_selectbox == "Online":
            st.info("Input data below")

            #Based on our optimal features selection
            st.subheader("Student informations:")
            gender = st.selectbox('Student Gender:', ('M', 'F'))
            NationalITy = st.selectbox('Student Nationality:', ('KW', 'lebanon', 'Egypt', 'SaudiArabia', 'USA', 'Jordan','venzuela', 'Iran', 'Tunis', 'Morocco', 'Syria', 'Palestine','Iraq', 'Lybia'))
            PlaceofBirth = st.selectbox('Place of Birth:', ('KuwaIT', 'lebanon', 'Egypt', 'SaudiArabia', 'USA', 'Jordan','venzuela', 'Iran', 'Tunis', 'Morocco', 'Syria', 'Iraq','Palestine', 'Lybia'))
            StageID = st.selectbox('Student Stage ID:', ('lowerlevel', 'MiddleSchool', 'HighSchool'))
            GradeID = st.selectbox('Student Grade ID:', ('G-04', 'G-07', 'G-08', 'G-06', 'G-05', 'G-09', 'G-12', 'G-11','G-10', 'G-02'))
            SectionID = st.selectbox('Student SectionID:', ('A', 'B', 'C'))
            Topic = st.selectbox('Which Subject:', ('IT', 'Math', 'Arabic', 'Science', 'English', 'Quran', 'Spanish','French', 'History', 'Biology', 'Chemistry', 'Geology'))
            Semester = st.selectbox('Which Semester:', ('F', 'S'))
            Relation = st.selectbox('who is a supervisor:', ('Father', 'Mum'))
            raisedhands = st.number_input('How many times he/she asked Question?',min_value=0, max_value=200, value=0)
            VisITedResources = st.number_input('How many times he/she visted online resources?',min_value=0, max_value=200, value=1)
            AnnouncementsView = st.number_input('How many times he/she viewed announcements?',min_value=0, max_value=200, value=1)
            Discussion = st.number_input('Number of discussion per semester:',min_value=0, max_value=200, value=1)
            ParentAnsweringSurvey = st.selectbox('Parent Answering Survey?', ('Yes', 'No'))
            ParentschoolSatisfaction = st.selectbox('Parent school Satisfaction:', ('Good', 'Bad'))
            StudentAbsenceDays = st.selectbox('Student Absence Days', ('Under-7', 'Above-7'))

            data = {
                    'gender':gender,
                    'NationalITy':NationalITy,
                    'PlaceofBirth':PlaceofBirth,
                    'StageID':StageID,
                    'GradeID':GradeID,
                    'SectionID':SectionID,
                    'Topic':Topic,
                    'Semester':Semester,
                    'Relation':Relation,
                    'raisedhands':raisedhands, 
                    'VisITedResources':VisITedResources, 
                    'AnnouncementsView':AnnouncementsView, 
                    'Discussion':Discussion, 
                    'ParentAnsweringSurvey':ParentAnsweringSurvey, 
                    'ParentschoolSatisfaction':ParentschoolSatisfaction, 
                    'StudentAbsenceDays':StudentAbsenceDays
                    }
            features_df = pd.DataFrame.from_dict([data])
            st.markdown("<h3></h3>", unsafe_allow_html=True)
            st.write('Overview of input is shown below')
            st.markdown("<h3></h3>", unsafe_allow_html=True)
            st.dataframe(features_df)

            #Preprocess inputs
            preprocess_df = preprocess_performance(features_df, 'Online')

            prediction = model.predict(preprocess_df)

            if st.button('Predict'):
                if prediction == 0:
                    st.warning('Nice! High performance........ keep it up')
                elif prediction == 1:
                    st.warning('Bad! Low performance.')
                else:
                    st.success('Wow! Medium Performance....... keep going')
                
        else:
            # Batch prediction
            st.subheader("Dataset upload")
            uploaded_file = st.file_uploader("Choose a file")
            if uploaded_file is not None:
                data = pd.read_csv(uploaded_file)
                #Get overview of data
                st.write(data.head())
                st.markdown("<h3></h3>", unsafe_allow_html=True)
                #Preprocess inputs
                preprocess_df = preprocess_performance(data, "Batch")
                if st.button('Predict'):
                    #Get batch prediction
                    prediction = model.predict(preprocess_df)
                    prediction_df = pd.DataFrame(prediction, columns=["Predictions"])
                    prediction_df = prediction_df.replace({1:'Low performance.',0:'High performance.',2:'Medium performance.'})

                    st.markdown("<h3></h3>", unsafe_allow_html=True)
                    st.subheader('Prediction')
                    st.write(prediction_df)
             
                
                        
        
    
    else:
        st.title('Student Dropout Prediction')
        st.text('Select a page in the sidebar')

        if add_selectbox == "Online":
            st.info("Input data below")
            #Based on our optimal features selection
            st.subheader("Student informations:")
            sem1kt = st.number_input('Semester 1 KT ? ',min_value=0, max_value=5, value=0)
            sem2sgpa = st.number_input('Semester  2 SGPA ?',min_value=0, max_value=5, value=0)
            sem2kt = st.number_input('Semester 2 KT ?',min_value=0, max_value=5, value=0)
            sem4sgpa = st.number_input('Semester 4 SGPA ? ',min_value=0, max_value=5, value=0)
            sem5sgpa = st.number_input('Semester  5 SGPA ?',min_value=0, max_value=5, value=0)
            sem6sgpa = st.number_input('Semester 6 SGPA ?',min_value=0, max_value=5, value=0)
            sem7sgpa = st.number_input('Semester 7 SGPA ?',min_value=0, max_value=5, value=0)
            sem8sgpa = st.number_input('Semester  8 SGPA ?',min_value=0, max_value=5, value=0)
           
            Hour_per_week_wriassignment = st.selectbox('How many hours per week do you spend on writing assignments?', ('1 - 5 hours', '5 - 10 hours', 'More than 10 hours','Less than 1 hour'))
            time_to_reach_college = st.selectbox('How much time does it take for you to reach college?', ('2 - 3 hours', '1 - 2 hours', 'Less than 1 hour','More than 3 hours'))
            Averageattendence = st.selectbox('How much would you consider as your average attendance throughout all the semesters so far?', ('70%  - 79%', '80%  - 89%', '90%  - 100%', '60%  - 69%','Less than 60%'))
            Internetathome = st.selectbox('Internet availability at home?', ('Yes', 'No'))
            hrstraightlecture = st.selectbox('Can you sit a lecture for 2 hrs straight? ', ('No', 'Yes'))
            submission_on_time = st.selectbox('Do you do your submissions on time?', ('Yes', 'No'))
            Five_lecture = st.selectbox('If there are 5 hrs of lectures per day, would you attend all?', ('Yes', 'No'))
            practical = st.selectbox('If there are 5 hrs of practicals per day, would you attend all? ', ('No', 'Yes'))
            Feedback = st.selectbox('Your teacher feedbacks', ('Good student', 'Good leadership skills', 'Hard working','Responsible', 'Disciplined and hard working',
                                     'Good but can be better', 'Excellent performance', 'willingness to put effort', 'Willingness to Put Forth Effort',
                                    'Respectful to Authority and Others','Solid Social and Emotional Skills', 'Self-Motivated',
                                     'Eagerness to Learn', 'Not attentive','Does not follow my lecture', 'Very talkative','Needs improvement', 'Poor attendance', 'Lagging Behind', 'Disappointed performance', 'Argues with teacher', 'Bunk lectuer','Always late', 'Always late and Does not follow my lecture','Does not follow my lecture and Very talkative','Very talkative and Poor attendance','Argues with teacher and Bunk lectuer','Needs improvement and Not attentive'))
            transportation = st.selectbox('What is your preferred mode of transportation to reach college?', ('Train', 'Private Vehicle', 'Bus'))
            coaching = st.selectbox('Have you enrolled for coaching classes?', ('No', 'Yes'))

            data = {
                    'SEM 1 KT':sem1kt,
                    'SEM 2 SGPA':sem2sgpa,
                    'SEM 2 KT':sem2kt,
                    'SEM 4 SGPA':sem4sgpa,
                    'SEM 5 SGPA':sem5sgpa,
                    'SEM 6 SGPA':sem6sgpa,
                    'SEM 7 SGPA':sem7sgpa,
                    'SEM 8 SGPA':sem8sgpa,
                    'Hour_per_week_wriassignment':Hour_per_week_wriassignment,
                    'time_to_reach college':time_to_reach_college, 
                    'Average attendence':Averageattendence, 
                    'Internet at home':Internetathome, 
                    '2 hr straight lecture':hrstraightlecture, 
                    'submission on time':submission_on_time, 
                    'Five lecture straight,woulf you attend all?':Five_lecture, 
                    'Five hr practical staight,do you attend all':practical,
                    'Feedback of teacher':Feedback,
                    'preffered transportatin to college':transportation, 
                    'Enrolled to coaching class':coaching, 
                    }
            features_df = pd.DataFrame.from_dict([data])
            st.markdown("<h3></h3>", unsafe_allow_html=True)
            st.write('Overview of input is shown below')
            st.markdown("<h3></h3>", unsafe_allow_html=True)
            st.dataframe(features_df)

            #Preprocess inputs
            preprocess_df = preprocess_dropout(features_df, 'Online')

            prediction = model2.predict(preprocess_df)

            if st.button('Predict'):
                if prediction == 1:
                    st.warning('Dropout')
                else:
                    st.success('Not Dropout')
                
        else:
            # Batch prediction
            st.subheader("Dataset upload")
            uploaded_file = st.file_uploader("Choose a file")
            if uploaded_file is not None:
                data = pd.read_csv(uploaded_file)
                #Get overview of data
                st.write(data.head())
                st.markdown("<h3></h3>", unsafe_allow_html=True)
                #Preprocess inputs
                preprocess_df = preprocess_dropout(data, "Batch")
                if st.button('Predict'):
                    #Get batch prediction
                    prediction = model2.predict(preprocess_df)
                    prediction_df = pd.DataFrame(prediction, columns=["Predictions"])
                    prediction_df = prediction_df.replace({1:'Dropout.', 2:'Not Drotout'})

                    st.markdown("<h3></h3>", unsafe_allow_html=True)
                    st.subheader('Prediction')
                    st.write(prediction_df)
       
 


         
   

if __name__ == '__main__':
        main()

