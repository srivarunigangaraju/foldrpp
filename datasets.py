from foldrpp import Classifier


def heart_disease():
    attrs = ['HighBP','HighChol','CholCheck','BMI','Smoke',
             'Stroke',	'Diabetes',	'PhysActivity',	'Fruits','Veggies',	'HvyAlcoholConsump',
             'AnyHealthcare','NoDocbcCost',	'GenHlth','MentHlth','PhysHlth',
             'DiffWalk','Sex',	'Age',	'Education','Income']
    nums = ['HighBP','HighChol','CholCheck','BMI','Smoke','Stroke',	'Diabetes',
            'PhysActivity',	'Fruits','Veggies',	'HvyAlcoholConsump',
             'AnyHealthcare','NoDocbcCost',	'GenHlth','MentHlth','PhysHlth',
             'DiffWalk','Age']
    model = Classifier(attrs=attrs, numeric=nums, label='HeartDiseaseorAttack', pos='1')
    data = model.load_data('data/heartdisease/heartdisease.csv')
    print('\n% heartdisease dataset', len(data), len(data[0]))
    return model, data


def heart_failure():
    attrs = ['age',	'anaemia',	'creatinine_phosphokinase',	'diabetes',
             'ejection_fraction',	'high_blood_pressure',	'platelets','serum_creatinine',
             'serum_sodium',	'sex',	'smoking',	'time']

    nums = ['age',	'anaemia',	'creatinine_phosphokinase',	'diabetes',
             'ejection_fraction',	'high_blood_pressure',	'platelets','serum_creatinine',
             'serum_sodium',	'sex',	'smoking',	'time']
    model = Classifier(attrs=attrs, numeric=nums, label='DEATH_EVENT', pos='1')
    data = model.load_data('data/heart_failure/heart_failure.csv')
    print('\n% heart failure dataset', len(data), len(data[0]))
    return model, data


def heart_values():
    attrs = ['slope_of_peak_exercise_st_segment'
             'resting_blood_pressure',
             'chest_pain_type',	'num_major_vessels'
             'fasting_blood_sugar_gt_120_mg_per_dl','resting_ekg_results',	'serum_cholesterol_mg_per_dl',
             'oldpeak_eq_st_depression','sex','age','max_heart_rate_achieved',	'exercise_induced_angina']

    nums = ['slope_of_peak_exercise_st_segment'
             'resting_blood_pressure',
        	'num_major_vessels'
             'fasting_blood_sugar_gt_120_mg_per_dl','resting_ekg_results',	'serum_cholesterol_mg_per_dl',
             'oldpeak_eq_st_depression','age',	'max_heart_rate_achieved']
    model = Classifier(attrs=attrs, numeric=nums, label='thal', pos='normal')
    data = model.load_data('data/heartvalues/heartvalues.csv')
    print('\n% heartvalues dataset', len(data), len(data[0]))
    return model, data


