import streamlit as st
import joblib
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from pathlib import Path



def app():
    st.markdown("<h1 style='text-align: center; color: cyan'>Fastest Lap Time Prediction</h1>", unsafe_allow_html=True)
    st.write("<h2 style= 'text-align: center; color: orange'>This app predicts the fastest lap time of a driver and his car based on the data you provide.</h1>", unsafe_allow_html=True)

    model_path = Path(__file__).parent / 'final_model.pkl'
    final_model = joblib.load(model_path)

    if 'predictions' not in st.session_state:
        st.session_state.predictions = []
    if 'page' not in st.session_state:
        st.session_state.page = 'home'

    if st.session_state.page == 'home':
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button('Custom Values'):
                st.session_state.page = 'custom'
        with col3:
            if st.button('Preset Values'):
                st.session_state.page = 'preset'
    
    if st.session_state.page == 'custom':

        with st.expander("Car Specifications"):
            Weight = st.number_input("Weight (kg)", value=1500)
            horsepower = st.number_input('Engine_Power_hp', value=500)
            torque = st.number_input("Torque", value=500)
            top_speed = st.number_input("Top Speed (km/h)", value=300)
            acceleration = st.number_input("0-100 km/h (seconds)", value=3)
            WeightDistribution = st.slider("Weight Distribution (%)", 0, 100, 50, format="%d%%")
            FrontTireDiameter = st.number_input("Front Tire Diameter (cm)", value=18)
            RearTireDiameter = st.number_input("Rear Tire Diameter (cm)", value=18)
            DragCoefficient = st.number_input("Drag Coefficient", value=0.3)
            FrontalArea = st.number_input("Frontal Area (m^2)", value=2)

        with st.expander("Tire and Suspension Setup"):
            FrontTirePressure = st.number_input("Front Tire Pressure (bar)", value=1.5)
            RearTirePressure = st.number_input("Rear Tire Pressure (bar)", value=1.5)
            FrontSuspension = st.number_input("Front Suspension (mm)", value=150)
            RearSuspension = st.number_input("Rear Suspension (mm)", value=150)
            FrontTireDegradation = st.number_input("Front Tire Degradation (%)", value=0)
            RearTireDegradation = st.number_input("Rear Tire Degradation (%)", value=0)
            TireChangeCount = st.number_input("Tire Change Count", value=0)

        with st.expander("Driver Characteristics"):
            DriverReflexes = st.number_input("Driver Reflexes (ms)", value=120)
            DriverExperience = st.selectbox("Driver Experience", options=["Beginner", "Intermediate", "Advanced", "Professional"])
            DriverFatigue = st.slider("Driver Fatigue (%)", 0, 100, 50, format="%d%%")
            DriverWeight = st.number_input("Driver Weight (kg)", value=70)
        
        with st.expander("Track Conditions"):
            TrackTemperature = st.number_input("Track Temperature (°C)", value=30)
            AmbiantTemperature = st.number_input("Ambiant Temperature (°C)", value=25)
            Humidity = st.slider("Humidity (%)", 0, 100, 50, format="%d%%")
            WindSpeed = st.number_input("Wind Speed (km/h)", value=0)
            WindDirection = st.number_input("Wind Direction (°)", value=0)
            TrackCondition = st.selectbox("Track Condition", options=["Dry", "Wet", "Damp"])
            TrackAltitude = st.number_input("Track Altitude (m)", value=0)
            TrackLength = st.number_input("Track Length (km)", value=5)
            SurfaceType = st.selectbox("Surface Type", options=["Asphalt", "Gravel", "Concrete"])
            MaxGradient = st.number_input("Max Gradient (%)", value=0)
            NumberCorner = st.number_input("Number of Corners", value=10)
            NumberStraight = st.number_input("Number of Straights", value=2)

        with st.expander("Brake and Fuel System"):
            FrontBrakeTemperature = st.number_input("Front Brake Temperature (°C)", value=300)
            RearBrakeTemperature = st.number_input("Rear Brake Temperature (°C)", value=300)
            FuelType = st.selectbox("Fuel Type", options=["Petrol", "Diesel", "Electric"])
            FuelConsumption = st.number_input("Fuel Consumption (L/100km)", value=10)

            
        with st.expander("Race and Strategy"):
            Strategy = st.selectbox("Strategy", options=["Aggressive", "Conservative"])
            Overtakes = st.number_input("Overtakes", value=0)
            HarshBraking = st.number_input("Harsh Braking", value=0)
            DriverMistakes = st.number_input("Driver Mistakes", value=0)
            TrajectoryChanges = st.number_input("Trajectory Changes", value=0)

        with st.expander("Maintenance and Technical Health"):
            GearboxCondition = st.selectbox("Gearbox Condition", options=["Good", "Acceptable", "Bad"])
            EngineCondition = st.selectbox("Engine Condition", options=["Good", "Acceptable", "Bad"])
            TechnicalProblems = st.selectbox("Technical Problems", options=["Yes", "No"])
            
            
        col1, col2, col3 = st.columns(3)
        with col2:
            if st.button('Predict Fastest Lap Time'):
                input_data = {
                    'Car_Weight_kg': Weight,
                    'Engine_Power_hp': horsepower,
                    'Max_Torque_Nm': torque,
                    'Top_Speed_kmh': top_speed,
                    'Acceleration_0_100_kmh': acceleration,
                    'Weight_Distribution_percentage': WeightDistribution,
                    'Transmission_Type': 0,  
                    'Gear_Count': 0,  
                    'Front_Tire_Pressure_bar': FrontTirePressure,
                    'Rear_Tire_Pressure_bar': RearTirePressure,
                    'Front_Tire_Diameter_cm': FrontTireDiameter,
                    'Rear_Tire_Diameter_cm': RearTireDiameter,
                    'Front_Suspension': FrontSuspension,
                    'Rear_Suspension': RearSuspension,
                    'Aerodynamics_Drag_Coefficient_Cd': DragCoefficient,
                    'Frontal_Area_m2': FrontalArea,
                    'Fuel_Type': FuelType,
                    'Fuel_Consumption_L_100km': FuelConsumption,
                    'Front_Brake_Temperature_C': FrontBrakeTemperature,
                    'Rear_Brake_Temperature_C': RearBrakeTemperature,
                    'Differential_Type': 0,  
                    'Front_Camber_Angle_deg': 0,  
                    'Rear_Camber_Angle_deg': 0,  
                    'Front_Brakes': 0,  
                    'Rear_Brakes': 0,  
                    'Track_Temperature_C': TrackTemperature,
                    'Ambient_Temperature_C': AmbiantTemperature,
                    'Humidity_percentage': Humidity,
                    'Wind_Speed_kmh': WindSpeed,
                    'Wind_Direction_deg': WindDirection,
                    'Track_Condition': TrackCondition,
                    'Track_Altitude_m': TrackAltitude,
                    'Track_Length_km': TrackLength,
                    'Surface_Type': SurfaceType,
                    'Number_of_Corners': NumberCorner,
                    'Number_of_Straights': NumberStraight,
                    'Max_Gradient_percentage': MaxGradient,
                    'Number_of_Laps': 0,  
                    'Driver_Reflexes_ms': DriverReflexes,
                    'Driver_Experience': DriverExperience,
                    'Driver_Fatigue': DriverFatigue,
                    'Driver_Weight_kg': DriverWeight,
                    'Race_Strategy': Strategy,
                    'Overtakes': Overtakes,
                    'Harsh_Braking_Count': HarshBraking,
                    'Driver_Mistakes': DriverMistakes,
                    'Average_Lap_Speed_kmh': 0,  
                    'Total_Race_Time_min': 0,  
                    'Fastest_Lap_Time_s': 0,  
                    'Slowest_Lap_Time_s': 0,  
                    'Front_Tire_Degradation_percentage': FrontTireDegradation,
                    'Rear_Tire_Degradation_percentage': RearTireDegradation,
                    'Tire_Changes_Count': TireChangeCount,
                    'Gearbox_Condition': GearboxCondition,
                    'Engine_Condition': EngineCondition,
                    'Technical_Problems': TechnicalProblems,
                    'Trajectory_Changes': TrajectoryChanges
                }
                le = LabelEncoder()
                
                categorical_features = ['Driver_Experience', 'Track_Condition', 'Surface_Type', 'Fuel_Type', 'Gearbox_Condition', 'Engine_Condition', 'Technical_Problems', 'Race_Strategy']
                for feature in categorical_features:
                    input_data[feature] = le.fit([input_data[feature]]).transform([input_data[feature]])[0]
                
                input_df = pd.DataFrame([input_data])

                
                prediction = final_model.predict(input_df)
                st.session_state.predictions.append(prediction)


    if st.session_state.page == 'preset':

        input_data = {
            'Car_Weight_kg': 0,
            'Engine_Power_hp': 0,
            'Max_Torque_Nm': 0,
            'Top_Speed_kmh': 0,
            'Acceleration_0_100_kmh': 0,
            'Weight_Distribution_percentage': 0,
            'Transmission_Type': 0,
            'Gear_Count': 0,
            'Front_Tire_Pressure_bar': 0,
            'Rear_Tire_Pressure_bar': 0,
            'Front_Tire_Diameter_cm': 0,
            'Rear_Tire_Diameter_cm': 0,
            'Front_Suspension': 0,
            'Rear_Suspension': 0,
            'Aerodynamics_Drag_Coefficient_Cd': 0,
            'Frontal_Area_m2': 0,
            'Fuel_Type': 0,
            'Fuel_Consumption_L_100km': 0,
            'Front_Brake_Temperature_C': 0,
            'Rear_Brake_Temperature_C': 0,
            'Differential_Type': 0,
            'Front_Camber_Angle_deg': 0,
            'Rear_Camber_Angle_deg': 0,
            'Front_Brakes': 0,
            'Rear_Brakes': 0,
            'Track_Temperature_C': 0,
            'Ambient_Temperature_C': 0,
            'Humidity_percentage': 0,
            'Wind_Speed_kmh': 0,
            'Wind_Direction_deg': 0,
            'Track_Condition': 0,
            'Track_Altitude_m': 0,
            'Track_Length_km': 0,
            'Surface_Type': 0,
            'Number_of_Corners': 0,
            'Number_of_Straights': 0,
            'Max_Gradient_percentage': 0,
            'Number_of_Laps': 0,
            'Driver_Reflexes_ms': 0,
            'Driver_Experience': 0,
            'Driver_Fatigue': 0,
            'Driver_Weight_kg': 0,
            'Race_Strategy': 0,
            'Overtakes': 0,
            'Harsh_Braking_Count': 0,
            'Driver_Mistakes': 0,
            'Average_Lap_Speed_kmh': 0,
            'Total_Race_Time_min': 0,
            'Fastest_Lap_Time_s': 0,
            'Slowest_Lap_Time_s': 0,
            'Front_Tire_Degradation_percentage': 0,
            'Rear_Tire_Degradation_percentage': 0,
            'Tire_Changes_Count': 0,
            'Gearbox_Condition': 0,
            'Engine_Condition': 0,
            'Technical_Problems': 0,
            'Trajectory_Changes': 0
        }

        car_presets = {
            "GT4": {
                'Car_Weight_kg': 1350,
                'Engine_Power_hp': 450,
                'Max_Torque_Nm': 480,
                'Top_Speed_kmh': 250,
                'Acceleration_0_100_kmh': 4.0,
                'Weight_Distribution_percentage': 50,
                'Front_Tire_Diameter_cm': 18,
                'Rear_Tire_Diameter_cm': 18,
                'Aerodynamics_Drag_Coefficient_Cd': 0.32,
                'Frontal_Area_m2': 2.0,
                'Front_Tire_Pressure_bar': 1.8,
                'Rear_Tire_Pressure_bar': 1.8,
                'Front_Suspension': 120,
                'Rear_Suspension': 120,
                'Fuel_Consumption_L_100km': 12,
            },
            "GT3": {
                'Car_Weight_kg': 1250,
                'Engine_Power_hp': 550,
                'Max_Torque_Nm': 520,
                'Top_Speed_kmh': 280,
                'Acceleration_0_100_kmh': 3.5,
                'Weight_Distribution_percentage': 55,
                'Front_Tire_Diameter_cm': 19,
                'Rear_Tire_Diameter_cm': 19,
                'Aerodynamics_Drag_Coefficient_Cd': 0.30,
                'Frontal_Area_m2': 1.9,
                'Front_Tire_Pressure_bar': 1.9,
                'Rear_Tire_Pressure_bar': 1.9,
                'Front_Suspension': 130,
                'Rear_Suspension': 130,
                'Fuel_Consumption_L_100km': 10,
            },
            "GT2": {
                'Car_Weight_kg': 1200,
                'Engine_Power_hp': 650,
                'Max_Torque_Nm': 600,
                'Top_Speed_kmh': 300,
                'Acceleration_0_100_kmh': 3.2,
                'Weight_Distribution_percentage': 56,
                'Front_Tire_Diameter_cm': 20,
                'Rear_Tire_Diameter_cm': 20,
                'Aerodynamics_Drag_Coefficient_Cd': 0.28,
                'Frontal_Area_m2': 1.8,
                'Front_Tire_Pressure_bar': 2.0,
                'Rear_Tire_Pressure_bar': 2.0,
                'Front_Suspension': 140,
                'Rear_Suspension': 140,
                'Fuel_Consumption_L_100km': 9,
            },
            "Hypercar": {
                'Car_Weight_kg': 1000,
                'Engine_Power_hp': 1000,
                'Max_Torque_Nm': 1000,
                'Top_Speed_kmh': 350,
                'Acceleration_0_100_kmh': 2.5,
                'Weight_Distribution_percentage': 58,
                'Front_Tire_Diameter_cm': 21,
                'Rear_Tire_Diameter_cm': 21,
                'Aerodynamics_Drag_Coefficient_Cd': 0.25,
                'Frontal_Area_m2': 1.7,
                'Front_Tire_Pressure_bar': 2.2,
                'Rear_Tire_Pressure_bar': 2.2,
                'Front_Suspension': 150,
                'Rear_Suspension': 150,
                'Fuel_Consumption_L_100km': 5,
            }
        }


        # Presets pour les pistes
        track_presets = {
            "Monza": {
                'Track_Temperature_C': 30,
                'Ambient_Temperature_C': 25,
                'Humidity_percentage': 40,
                'Wind_Speed_kmh': 5,
                'Wind_Direction_deg': 180,
                'Track_Condition': 0,
                'Track_Altitude_m': 160,
                'Track_Length_km': 5.8,
                'Surface_Type': 0,
                'Max_Gradient_percentage': 3,
                'Number_of_Corners': 11,
                'Number_of_Straights': 4,
            },
            "LeMans": {
                'Track_Temperature_C': 25,
                'Ambient_Temperature_C': 20,
                'Humidity_percentage': 60,
                'Wind_Speed_kmh': 10,
                'Wind_Direction_deg': 90,
                'Track_Condition': 0,
                'Track_Altitude_m': 50,
                'Track_Length_km': 13.6,
                'Surface_Type': 0,
                'Max_Gradient_percentage': 2,
                'Number_of_Corners': 33,
                'Number_of_Straights': 3,
            },
            "Nurburgring": {
                'Track_Temperature_C': 18,
                'Ambient_Temperature_C': 15,
                'Humidity_percentage': 75,
                'Wind_Speed_kmh': 15,
                'Wind_Direction_deg': 270,
                'Track_Condition': 0,
                'Track_Altitude_m': 620,
                'Track_Length_km': 20.8,
                'Surface_Type': 0,
                'Max_Gradient_percentage': 10,
                'Number_of_Corners': 154,
                'Number_of_Straights': 5,
            },
            "Spa-Francorchamps": {
                'Track_Temperature_C': 22,
                'Ambient_Temperature_C': 20,
                'Humidity_percentage': 50,
                'Wind_Speed_kmh': 8,
                'Wind_Direction_deg': 0,
                'Track_Condition': 0,
                'Track_Altitude_m': 470,
                'Track_Length_km': 7.0,
                'Surface_Type': 0,
                'Max_Gradient_percentage': 6,
                'Number_of_Corners': 20,
                'Number_of_Straights': 3,
            }
        }


        col1, col2 = st.columns(2)
        car_name = None
        track_name = None
        with col1:
            track_name = st.selectbox('Track Presets', options=list(track_presets.keys()))
        with col2:
            car_name = st.selectbox('Car Presets', options=list(car_presets.keys()))

        selected_car = car_presets.get(car_name, {})
        selected_track = track_presets.get(track_name, {})
        
        
        input_data.update(selected_car)
        input_data.update(selected_track)
        
        input_df = pd.DataFrame([input_data])
        prediction = final_model.predict(input_df)
        st.session_state.predictions.append(prediction)

            
        col1, col2, col3 = st.columns(3)
        with col2:
            if st.button('Predict Fastest Lap Time'):
                le = LabelEncoder()
                
                categorical_features = ['Driver_Experience', 'Track_Condition', 'Surface_Type', 'Fuel_Type', 'Gearbox_Condition', 'Engine_Condition', 'Technical_Problems', 'Race_Strategy']
                for feature in categorical_features:
                    input_data[feature] = le.fit([input_data[feature]]).transform([input_data[feature]])[0]
                
                input_df = pd.DataFrame([input_data])

                
                prediction = final_model.predict(input_df)
                st.session_state.predictions.append(prediction)

    

    if len(st.session_state.predictions) > 0:
        def format_time(seconds):
            minutes = int(seconds // 60)
            remaining_seconds = seconds % 60
            whole_seconds = int(remaining_seconds)
            milliseconds = int((remaining_seconds - whole_seconds) * 1000)
            return f"{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


        prediction_data = {
            "Fastest Lap Time":  [format_time(pred) for pred in st.session_state.predictions],
            "Exact Time (seconds)": st.session_state.predictions,
            "Difference (seconds)": [0] + [
            st.session_state.predictions[i] - st.session_state.predictions[i - 1]
            for i in range(1, len(st.session_state.predictions))
            ]
        }
        prediction_df = pd.DataFrame(prediction_data)
        st.subheader('Prediction History')
        st.table(prediction_df)


    
    


    

if __name__ == '__main__':
    app()
