"""Generate sample Synthea-like data for testing the pipeline."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import uuid


def generate_patients(n_patients=50):
    """Generate sample patients data."""
    np.random.seed(42)
    
    patients = []
    for i in range(n_patients):
        # Generate birth dates for adults (18-80 years old)
        age = np.random.randint(18, 81)
        birth_date = datetime(2024, 12, 31) - timedelta(days=age*365 + np.random.randint(0, 365))
        
        patient = {
            'Id': f'patient_{i+1:03d}',
            'BirthDate': birth_date.strftime('%Y-%m-%d'),
            'DeathDate': '',  # Most patients alive
            'SSN': f'{np.random.randint(100, 999)}-{np.random.randint(10, 99)}-{np.random.randint(1000, 9999)}',
            'Drivers': '',
            'Passport': '',
            'Prefix': np.random.choice(['Mr.', 'Mrs.', 'Ms.'], p=[0.4, 0.3, 0.3]),
            'First': f'FirstName{i+1}',
            'Last': f'LastName{i+1}',
            'Suffix': '',
            'Maiden': '',
            'Marital': np.random.choice(['M', 'S'], p=[0.6, 0.4]),
            'Race': np.random.choice(['white', 'black', 'asian'], p=[0.7, 0.2, 0.1]),
            'Ethnicity': np.random.choice(['nonhispanic', 'hispanic'], p=[0.8, 0.2]),
            'Gender': np.random.choice(['M', 'F'], p=[0.5, 0.5]),
            'Birthplace': 'Boston, MA, US',
            'Address': f'{np.random.randint(1, 999)} Main St',
            'City': 'Boston',
            'State': 'Massachusetts',
            'County': 'Suffolk County',
            'Zip': '02101',
            'Lat': 42.3601,
            'Lon': -71.0589,
            'Healthcare_Expenses': np.random.randint(1000, 50000),
            'Healthcare_Coverage': np.random.randint(500, 25000)
        }
        patients.append(patient)
    
    return pd.DataFrame(patients)


def generate_conditions(patients_df):
    """Generate sample conditions data with diabetes."""
    np.random.seed(42)
    
    conditions = []
    
    # Ensure all patients have diabetes (SNOMED: 44054006)
    for _, patient in patients_df.iterrows():
        patient_id = patient['Id']
        
        # Diabetes condition
        start_date = datetime.strptime(patient['BirthDate'], '%Y-%m-%d') + timedelta(days=np.random.randint(365*20, 365*50))
        
        condition = {
            'Start': start_date.strftime('%Y-%m-%d'),
            'Stop': '',  # Chronic condition
            'Patient': patient_id,
            'Encounter': f'enc_{uuid.uuid4().hex[:8]}',
            'Code': '44054006',  # Diabetes mellitus SNOMED code
            'Description': 'Diabetes mellitus',
            'System': 'SNOMED-CT'
        }
        conditions.append(condition)
        
        # Add some other random conditions
        other_conditions = [
            ('38341003', 'Hypertension'),
            ('55822004', 'Hyperlipidemia'),
            ('195967001', 'Asthma')
        ]
        
        for code, desc in other_conditions:
            if np.random.random() < 0.3:  # 30% chance of having each condition
                cond_start = start_date + timedelta(days=np.random.randint(0, 365*5))
                condition = {
                    'Start': cond_start.strftime('%Y-%m-%d'),
                    'Stop': '',
                    'Patient': patient_id,
                    'Encounter': f'enc_{uuid.uuid4().hex[:8]}',
                    'Code': code,
                    'Description': desc,
                    'System': 'SNOMED-CT'
                }
                conditions.append(condition)
    
    return pd.DataFrame(conditions)


def generate_encounters(patients_df):
    """Generate sample encounters data."""
    np.random.seed(42)
    
    encounters = []
    encounter_classes = ['outpatient', 'emergency', 'inpatient', 'urgentcare', 'wellness']
    
    for _, patient in patients_df.iterrows():
        patient_id = patient['Id']
        
        # Generate encounters over the past 2 years
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2024, 12, 31)
        
        # Each patient has 5-20 encounters
        n_encounters = np.random.randint(5, 21)
        
        for i in range(n_encounters):
            enc_start = start_date + timedelta(days=np.random.randint(0, (end_date - start_date).days))
            enc_stop = enc_start + timedelta(hours=np.random.randint(1, 48))
            
            encounter = {
                'Id': f'enc_{uuid.uuid4().hex[:8]}',
                'Start': enc_start.strftime('%Y-%m-%d %H:%M:%S'),
                'Stop': enc_stop.strftime('%Y-%m-%d %H:%M:%S'),
                'Patient': patient_id,
                'Organization': 'Sample Hospital',
                'Provider': f'provider_{np.random.randint(1, 10)}',
                'Payer': 'Sample Insurance',
                'EncounterClass': np.random.choice(encounter_classes, p=[0.6, 0.1, 0.1, 0.1, 0.1]),
                'Code': '185349003',
                'Description': 'Encounter for check up',
                'Base_Encounter_Cost': np.random.randint(100, 2000),
                'Total_Claim_Cost': np.random.randint(100, 2000),
                'Payer_Coverage': np.random.randint(50, 1500),
                'ReasonCode': '',
                'ReasonDescription': ''
            }
            encounters.append(encounter)
    
    return pd.DataFrame(encounters)


def generate_observations(patients_df):
    """Generate sample observations data."""
    np.random.seed(42)
    
    observations = []
    
    # Define observation types
    obs_types = [
        ('4548-4', 'Hemoglobin A1c/Hemoglobin.total in Blood', '%'),
        ('17856-6', 'Hemoglobin A1c/Hemoglobin.total in Blood by HPLC', '%'),
        ('8480-6', 'Systolic blood pressure', 'mmHg'),
        ('8462-4', 'Diastolic blood pressure', 'mmHg'),
        ('29463-7', 'Body Weight', 'kg'),
        ('8302-2', 'Body Height', 'cm'),
        ('39156-5', 'Body Mass Index', 'kg/m2')
    ]
    
    for _, patient in patients_df.iterrows():
        patient_id = patient['Id']
        
        # Generate observations over the past 2 years
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2024, 12, 31)
        
        for code, description, unit in obs_types:
            # Each observation type has 3-10 measurements
            n_obs = np.random.randint(3, 11)
            
            for i in range(n_obs):
                obs_date = start_date + timedelta(days=np.random.randint(0, (end_date - start_date).days))
                
                # Generate realistic values based on observation type
                if 'A1c' in description:
                    value = np.random.normal(7.5, 1.2)  # Diabetic range
                    value = max(4.0, min(15.0, value))  # Reasonable bounds
                elif 'Systolic' in description:
                    value = np.random.normal(140, 20)
                    value = max(80, min(200, value))
                elif 'Diastolic' in description:
                    value = np.random.normal(85, 15)
                    value = max(50, min(120, value))
                elif 'Weight' in description:
                    value = np.random.normal(80, 15)
                    value = max(40, min(150, value))
                elif 'Height' in description:
                    value = np.random.normal(170, 10)
                    value = max(140, min(200, value))
                elif 'BMI' in description:
                    value = np.random.normal(28, 5)
                    value = max(15, min(50, value))
                else:
                    value = np.random.normal(100, 20)
                
                observation = {
                    'Date': obs_date.strftime('%Y-%m-%d'),
                    'Patient': patient_id,
                    'Encounter': f'enc_{uuid.uuid4().hex[:8]}',
                    'Category': 'vital-signs',
                    'Code': code,
                    'Description': description,
                    'Value': f'{value:.1f}',
                    'Units': unit,
                    'Type': 'numeric'
                }
                observations.append(observation)
    
    return pd.DataFrame(observations)


def generate_medications(patients_df):
    """Generate sample medications data."""
    np.random.seed(42)
    
    medications = []
    
    # Common diabetes medications
    diabetes_meds = [
        ('860975', 'Metformin 500 MG Oral Tablet'),
        ('860901', 'Insulin, Regular, Human 100 UNT/ML Injectable Solution'),
        ('897122', 'Glipizide 5 MG Oral Tablet'),
        ('896188', 'Lisinopril 10 MG Oral Tablet')
    ]
    
    for _, patient in patients_df.iterrows():
        patient_id = patient['Id']
        
        # Each patient has 2-8 medications
        n_meds = np.random.randint(2, 9)
        
        for i in range(n_meds):
            code, description = diabetes_meds[i % len(diabetes_meds)]
            
            start_date = datetime(2023, 1, 1) + timedelta(days=np.random.randint(0, 365))
            
            # Some medications are ongoing, others have stop dates
            if np.random.random() < 0.7:  # 70% ongoing
                stop_date = ''
            else:
                stop_date = (start_date + timedelta(days=np.random.randint(30, 365))).strftime('%Y-%m-%d')
            
            medication = {
                'Start': start_date.strftime('%Y-%m-%d'),
                'Stop': stop_date,
                'Patient': patient_id,
                'Payer': 'Sample Insurance',
                'Encounter': f'enc_{uuid.uuid4().hex[:8]}',
                'Code': code,
                'Description': description,
                'Base_Cost': np.random.randint(10, 200),
                'Payer_Coverage': np.random.randint(5, 150),
                'Dispenses': np.random.randint(1, 12),
                'TotalCost': np.random.randint(50, 1000),
                'ReasonCode': '44054006',
                'ReasonDescription': 'Diabetes mellitus'
            }
            medications.append(medication)
    
    return pd.DataFrame(medications)


def main():
    """Generate all sample data files."""
    print("Generating sample Synthea-like data...")
    
    # Create output directory
    output_dir = 'sample_data'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate data
    print("Generating patients...")
    patients_df = generate_patients(50)
    patients_df.to_csv(os.path.join(output_dir, 'patients.csv'), index=False)
    
    print("Generating conditions...")
    conditions_df = generate_conditions(patients_df)
    conditions_df.to_csv(os.path.join(output_dir, 'conditions.csv'), index=False)
    
    print("Generating encounters...")
    encounters_df = generate_encounters(patients_df)
    encounters_df.to_csv(os.path.join(output_dir, 'encounters.csv'), index=False)
    
    print("Generating observations...")
    observations_df = generate_observations(patients_df)
    observations_df.to_csv(os.path.join(output_dir, 'observations.csv'), index=False)
    
    print("Generating medications...")
    medications_df = generate_medications(patients_df)
    medications_df.to_csv(os.path.join(output_dir, 'medications.csv'), index=False)
    
    print(f"Sample data generated successfully in {output_dir}/")
    print(f"Patients: {len(patients_df)}")
    print(f"Conditions: {len(conditions_df)}")
    print(f"Encounters: {len(encounters_df)}")
    print(f"Observations: {len(observations_df)}")
    print(f"Medications: {len(medications_df)}")


if __name__ == "__main__":
    main()
