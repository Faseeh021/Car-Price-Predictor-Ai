import pandas as pd 
import numpy as np 
import pickle as pk 
import streamlit as st

model = pk.load(open('model.pkl', 'rb'))

st.header('Car Price Predictor Ai')

cars_data = pd.read_csv('olx_car_data_csv.csv')

def get_brand_name(car_name):
    if isinstance(car_name, str):
        car_name = car_name.split(' ')[0]
        return car_name.strip()
    return car_name

# Convert all values in the 'Brand' column to strings
cars_data['Brand'] = cars_data['Brand'].astype(str).apply(get_brand_name)

Brand = st.selectbox('Select Car Brand', cars_data['Brand'].unique())
Year = st.slider('Car Manufactured Year', 1994, 2024)
km_driven = st.slider('No of kms Driven', 11, 200000)
Fuel = st.selectbox('Fuel type', cars_data['Fuel'].unique())
Condition = st.selectbox('Condition', cars_data['Condition'].unique())
Model = st.selectbox('Model', cars_data['Model'].unique())
Registered_City = st.selectbox('Registered City', cars_data['Registered City'].unique())
Transaction_Type = st.selectbox('Transaction Type', cars_data['Transaction Type'].unique())
Engine = st.selectbox('engine', cars_data['engine'].unique())

if st.button("Predict"):
    input_data_model = pd.DataFrame(
        [[Brand, Condition, Fuel, km_driven, Model, Registered_City, Transaction_Type, Year, Engine]],
        columns=['Brand', 'Condition', 'Fuel', 'KMs Driven', 'Model', 'Registered City', 'Transaction Type', 'Year','engine']
    )
    
    input_data_model['Brand'] = input_data_model['Brand'].replace(
        ['Toyota', 'Suzuki', 'Honda', 'Daihatsu', 'Mitsubishi', 'KIA', 'Other', 'Nissan', 'BMW', 'Mazda', 'Chevrolet', 'Daewoo', 
         'Hyundai', 'FAW', 'Mercedes', 'Classic', 'Lexus', 'Audi', 'Range', 'Changan', 'Porsche'],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    ).infer_objects(copy=False)
    input_data_model['Condition'] = input_data_model['Condition'].replace(['Used', 'New'], [1, 2]).infer_objects(copy=False)
    input_data_model['Fuel'] = input_data_model['Fuel'].replace(
        ['Diesel', 'Petrol', 'CNG', 'Hybrid', 'LPG'], [1, 2, 3, 4, 5]).infer_objects(copy=False)
    input_data_model['Model'] = input_data_model['Model'].replace(
        ['Prado', 'Bolan', 'Alto', 'Corolla XLI', 'Corrolla Altis', 'Cultus VXL', 'Civic VTi', 'Khyber', 'Liana', 'Passo',
         'Civic Prosmetic', 'Civic EXi', 'Charade', 'Pajero Mini', 'Margalla', 'City IVTEC', 'Classic', 'Other', 'Corolla GLI',
         'Cultus VXR', 'Dayz Highway Star', 'Mehran VX', 'Vitz', 'Mehran VXR', 'Carry', 'Cultus VX', 'Baleno', 'Mira',
         'Civic VTi Oriel Prosmatec', 'Cuore', 'Corolla 2.0 D', 'Corolla XE', 'Surf', 'FX', 'City IDSI', 'Premio', 'Sprinter',
         '3 Series', 'Hilux', 'Lancer', 'Swift', 'Estima', 'Vamos', 'Starlet', 'Prius', 'Joy', '323', 'Racer', 'Sunny', 'Accord',
         'Zest', 'AD Van', 'APV', 'March', 'Pride', 'Sportage', 'Terios Kid', 'Santro', 'Fit', 'V2', 'City Aspire',
         'Civic VTi Oriel', 'BR-V', 'Kei', 'Probox', 'Hijet', '86', 'Yaris', 'Aygo', 'Rush', 'Dayz', 'Fortuner', 'Every', 'Camry', 'Aqua',
         'Corolla Assista', 'Sera', 'Acty', 'Moco', 'Wagon R', 'Smart', 'Demio', 'Corona', 'Every Wagon', 'Civic Hybrid', 'Duet', 'Vezel',
         'Corolla Axio', 'City Vario', 'Corolla Fielder', 'Palette Sw', 'Platz', 'Lancer Evolution', 'Move', 'Gx Series', '5 Series',
         '240 Gd', 'Mira Cocoa', 'Wingroad', 'Belta', 'Atrai Wagon', 'Minicab Bravo', 'Allion', 'Blue Bird', 'Exclusive', 'N Wgn',
         'CT200h', 'Spectra', 'Beat', 'Mark II', 'Airwave', 'Supra', 'Ek Wagon', 'EK Custom', 'Lite Ace', '120 Y', 'Harrier',
         'Pixis Epoch', '626', 'Galant', 'Sonica', 'Carol Eco', 'Accent', 'IST', 'Land Cruiser', 'Palette', 'Juke', 'Potohar', 'CR-V',
         'Clipper', 'Sirius', 'Roox', '350Z', 'Pajero', 'Vitara', 'Alto Lapin', 'Wagon R Stingray', 'Prius Alpha', 'Minica', 'Cast',
         'Carol', 'Hustler', 'S Class', 'Boon', 'Jimny', 'Fj Cruiser', 'X-PV', 'Esse', 'Mirage', 'RX8', 'Carisma', 'iQ', 'Ek Sport',
         'CLK Class', 'Note', 'Stream', 'Carrier', 'Town Ace', 'I', 'N Box', 'Excel', 'B2200', 'Tundra', 'Scrum', 'Celica', 'Van', 'C Class',
         'Pickup', 'Kizashi', 'Freed', 'RX Series', 'Life', 'Otti', 'Spacia', 'Cervo', 'X5 Series', '200 D', 'Patrol', 'Acura',
         'Crown', 'Copen', 'Scrum Wagon', 'Charmant', 'Mark X', 'Spike', 'Vogue', 'Axela', '929', 'Ravi', 'Noah', 'Jade Hybrid', 'Tanto',
         'X Trail', 'EK Space Custom', 'Cielo', 'I Mivec', 'N One', 'i8', 'S660', 'Tiida', 'CR-Z', 'Verossa', 'Flair Wagon', 'Grace Hybrid',
         'E Class', 'Wish', 'L300', 'Auris', 'Cayenne', 'Silverado', 'Insight', 'Thats', 'C-HR', 'Familia Van', 'Jimny Sierra',
         'D Series', 'Escudo', 'Vanette', 'Rvr', 'LX Series', '250 D', 'Cami', 'Bluebird Sylphy', 'Pino', 'Ignis', 'A4', 'Wake', 'Spark',
         'Sirion', 'Q7', 'Zest Spark', 'Cruze', 'Avanza', 'Flair', 'ISIS', 'Cross Road', '7 Series', 'Matiz', 'Pulsar', 'Ciaz', 'A6',
         'Sonata', '6 Series', 'Cressida', 'Solio', 'Alphard Hybrid', 'Azwagon', 'Rav4', 'Toyo Ace'],
        [i for i in range(1, 250)]
    ).infer_objects(copy=False)
    input_data_model['Registered City'] = input_data_model['Registered City'].replace(
        ['Karachi', 'Hyderabad', 'Bagh', 'Sukkar', 'Bahawalnagar', 'Lahore', 'Askoley', 'Khanpur', 'Quetta', 'Karak', 
         'Islamabad', 'Sialkot', 'Pakpattan', 'Lasbela', 'Sukkur', 'Rawalpindi', 'Bahawalpur', 'Ali Masjid', 'Multan', 
         'Khaplu', 'Tank', 'Badin', 'Rahimyar Khan', 'Chilas', 'Kasur', 'Khushab', 'Vehari', 'Chitral', 'Khanewal', 
         'Attock', 'Larkana', 'Bela', 'Khairpur'],
        list(range(1, 34))
    ).infer_objects(copy=False)
    input_data_model['Transaction Type'] = input_data_model['Transaction Type'].replace(
        ['Cash', 'Installment/Leasing'],
        [1, 2]
    ).infer_objects(copy=False)
    input_data_model['engine'] = input_data_model['engine'].replace(
    ['1248 CC', '1498 CC', '1497 CC', '1396 CC', '1298 CC', '1197 CC',
       '1061 CC', '796 CC', '1364 CC', '1399 CC', '1461 CC', '993 CC',
       '1198 CC', '1199 CC', '998 CC', '1591 CC', '2179 CC', '1368 CC',
       '2982 CC', '2494 CC', '2143 CC', '2477 CC', '1462 CC', '2755 CC',
       '1968 CC', '1798 CC', '1196 CC', '1373 CC', '1598 CC', '1998 CC',
       '1086 CC', '1194 CC', '1172 CC', '1405 CC', '1582 CC', '999 CC',
       '2487 CC', '1999 CC', '3604 CC', '2987 CC', '1995 CC', '1451 CC',
       '1969 CC', '2967 CC', '2497 CC', '1797 CC', '1991 CC', '2362 CC',
       '1493 CC', '1599 CC', '1341 CC', '1794 CC', '799 CC', '1193 CC',
       '2696 CC', '1495 CC', '1186 CC', '1047 CC', '2498 CC', '2956 CC',
       '2523 CC', '1120 CC', '624 CC', '1496 CC', '1984 CC', '2354 CC',
       '814 CC', '793 CC', '1799 CC', '936 CC', '1956 CC', '1997 CC',
       '1499 CC', '1948 CC', '2997 CC', '2489 CC', '2499 CC', '2609 CC',
       '2953 CC', '1150 CC', '1994 CC', '1388 CC', '1527 CC', '2199 CC',
       '995 CC', '2993 CC', '1586 CC', '1390 CC', '909 CC', '2393 CC',
       '3198 CC', '1339 CC', '2835 CC', '2092 CC', '1595 CC', '2496 CC',
       '1596 CC', '1597 CC', '2596 CC', '2148 CC', '1299 CC', '1590 CC',
       '2231 CC', '2694 CC', '2200 CC', '1795 CC', '1896 CC', '1796 CC',
       '1422 CC', '1489 CC', '2197 CC', '2999 CC', '2650 CC', '1781 CC',
       '2359 CC', '1343 CC', '2446 CC', '3498 CC', '2198 CC', '1950 CC'],
    list(range(1, 121))
).infer_objects(copy=False)

    car_price = model.predict(input_data_model)

    st.markdown('Car Price is going to be '+str(car_price[0]))