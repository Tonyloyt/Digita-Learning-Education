from numpy.lib.function_base import average
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import pandas as pd
import numpy as np
import pickle
import joblib

lb_hpw=pickle.load(open('hour_per_week_assignment_lb.pkl','rb'))
lb_ttrc=pickle.load(open('time_to_reach_collg_lb.pkl','rb'))
lb_avgatt=pickle.load(open('average_attendence_lb.pkl','rb'))
lb_inter=pickle.load(open('Internet at home_lb.pkl','rb'))
lb_lecturetime=pickle.load(open('2_hr_lecture_lb.pkl','rb'))
lb_submission=pickle.load(open('submission_lb.pkl','rb'))
lb_5lect=pickle.load(open('5_hr_lecture_lb.pkl','rb'))
lb_5prac=pickle.load(open('5_hr_practical_lb.pkl','rb'))
lb_teacherfeedback=pickle.load(open('teacher_feedback_lb.pkl','rb'))
lb_preff_transport=pickle.load(open('preffered_transport_lb.pkl','rb'))
lb_enrolled_coaching=pickle.load(open('enrolled_coaching_lb.pkl','rb'))

def predict(data):
    clf = joblib.load('model.pkl','rb')
    pred = clf.predict(data)
    print(pred[0])
    if pred[0]==1:
        prediction = "Not a dropout"
    else:
        prediction = "Dropout"
   
    return prediction
    
data=[]
sem1kt=float(input("Enter semester 1 KT"))
data.append(sem1kt)
sem2sgpa=float(input("Enter semester 2 SGPA"))
data.append(sem2sgpa)
sem2kt=float(input("Enter semester 2 KT"))
data.append(sem2kt)
sem4sgpa=float(input("Enter semester 4 SGPA"))
data.append(sem4sgpa)
sem5sgpa=float(input("Enter semester 5 SGPA"))
data.append(sem5sgpa)
sem6sgpa=float(input("Enter semester 6 SGPA"))
data.append(sem6sgpa)
sem7sgpa=float(input("Enter semester 7 SGPA"))
data.append(sem7sgpa)
sem8sgpa=float(input("Enter semester 8 SGPA"))
data.append(sem8sgpa)
hrs_week=input("Hours per week?") # No. hours per week on assignment
data.append(lb_hpw.transform([hrs_week])[0])
time_to_collg=input("Time to reach college?") # time to reach collg
data.append(lb_ttrc.transform([time_to_collg])[0])
average_att=input("average attendence?") # Average attendence
data.append(lb_avgatt.transform([average_att])[0])
internet_home=input("Internet at home?") #  Internet at home
data.append(lb_inter.transform([internet_home])[0])
lecture_time=input("2 hr lecture time?") #  2 Hr staright lecture
data.append(lb_lecturetime.transform([lecture_time])[0])
submission=input("Submission on time?") #  Submission on time
data.append(lb_submission.transform([submission])[0])
five_lecture_straight=input("Five lecture straight") #  five hour lecture straight
data.append(lb_5lect.transform([five_lecture_straight])[0])
five_prac_straight=input("Five hr practical straight") #  five hour practical straight
data.append(lb_5prac.transform([five_prac_straight])[0])
teacher_fedback=input("Teacher feedback") #  Teacher feedback
data.append(lb_teacherfeedback.transform([teacher_fedback])[0])
preffered_transport=input("preffered_transport?") #  Preffered transport
data.append(lb_preff_transport.transform([preffered_transport])[0])
coaching_class=input("Enrolled to coaching?") #  Enrolled to caoching class
data.append(lb_enrolled_coaching.transform([coaching_class])[0])


data=np.asarray([data])

predict(data)



