# Importing neccesary Libraries 
import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')


def main():
    st.title("Project:- Sales Prediction by Machine Learning")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Video Games Sales Predictor ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)


    st.subheader("Performing following Activities: ")
    st.subheader("1.EDA 2.Data Visualization 3.Model Creation")
    if (st.button("Project Details")):
        st.write("""We are going to build project on The Machine Learning model to predict future sales of Video Games in gloabal market.
                We will learn different patterns and relationships between the input Variables or factors that affecting to the video games sales.
                We analyse data to get meaning information from them and Visualize them to recognise different patterns among them and showing meaning full information
                that will help us to predict sales data by creating effective machine learning model.
                We use Data set which contains the list of video games with sales greater than 16500 games in global market.""")
    
    if (st.button("Data set Details")):
        st.write("""
        Rank - Ranking of overall sales

        Name - The games name

        Platform - Platform of the games release (i.e. PC,PS4, etc.)

        Year - Year of the game's release

        Genre - Genre of the game

        Publisher - Publisher of the game

        NA_Sales - Sales in North America (in millions)

        EU_Sales - Sales in Europe (in millions)

        JP_Sales - Sales in Japan (in millions)

        Other_Sales - Sales in the rest of the world (in millions)

        Global_Sales - Total worldwide sales.""")

    activity=["Descriptive Analysis","Data Visualization","Sales Prediction"]
    selectact= st.sidebar.selectbox("Choose Analysis Method",activity)
    if(selectact=="Descriptive Analysis"):
        st.subheader("Exploratory Data Analysis")
        df = pd.read_csv("vgsales.csv")
        if(st.checkbox("Preview Dataset")):
            number = st.number_input("Select Number of Rows to View", value=1)
            st.write(df.head(number))
        if(st.checkbox("Shape of Dataset")):
            st.write(df.shape)
            data_dim = st.radio("Show Dimension By ", ("Rows","Columns","Size"))
            if(data_dim == "Rows"):
                st.text("Showing the Rows")
                st.write(df.shape[0])
            if(data_dim=="Columns"):
                st.text("Showing the Columns")
                st.write(df.shape[1])
            if(data_dim=="Size"):
                st.text("Showing the Size")
                st.write(df.size)
                
        if(st.checkbox("Select Columns")):
            all_columns = df.columns
            selected_columns = st.multiselect("Select Columns",all_columns)
            new_df = df[selected_columns]
            st.dataframe(new_df)
            
        if(st.button("Descriptive summary")):
            st.write(df.describe())

        if(st.button("Info summary")):
            info=""" RangeIndex: 16598 entries, 0 to 16597
Data columns (total 11 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   Rank          16598 non-null  int64  
 1   Name          16598 non-null  object 
 2   Platform      16598 non-null  object 
 3   Year          16327 non-null  float64
 4   Genre         16598 non-null  object 
 5   Publisher     16540 non-null  object 
 6   NA_Sales      16598 non-null  float64
 7   EU_Sales      16598 non-null  float64
 8   JP_Sales      16598 non-null  float64
 9   Other_Sales   16598 non-null  float64
 10  Global_Sales  16598 non-null  float64
dtypes: float64(6), int64(1), object(4)"""
            st.text("info of data set:{}".format(info))
        if(st.button("Unique values")):
            for col in df:
                uni=df[col].nunique()
                st.text("{} :- {}".format(col,uni))
        if(st.checkbox("Top most classes in categorical Features")):
            df_cat = df.select_dtypes('object')
            all_columns = df_cat.columns
            selected_columns = st.selectbox("Select Columns",all_columns)
            df2=df[selected_columns]
            topval=df2.value_counts().head()
            st.write(selected_columns)
            st.write(topval)

        st.subheader("The game which has highest global sales")
        if(st.button("Top sold game")):
            high = df["Global_Sales"].max()
            top_game = df[df["Global_Sales"] == high]
            st.write(top_game)

        st.subheader("The game which has Lowest global sales")
        if(st.button("less sold game")):
            low = df["Global_Sales"].min()
            low_game = df[df["Global_Sales"] == low]
            st.write(low_game)

        st.subheader("Top 10 publishers")
        if(st.button("publishers")):
            pub=df['Publisher'].value_counts().head(10)
            st.write(pub)

    if(selectact=="Data Visualization"):
        st.subheader("Data Visualizaton")
        df = pd.read_csv("vgsales.csv")

        if(st.checkbox("Overall visualization")):
            st.subheader("Using Pairplot Graph")
            fig = plt.figure()
            sns.pairplot(df)
            plt.show()
            st.pyplot(fig)

        if(st.sidebar.checkbox("Correlation Visualization")):
            st.subheader("Correlation between independent variable and target variable(Global Sales)")
            fig = plt.figure()
            sns.heatmap(df.corr(),annot=True,cmap='RdYlBu_r')
            plt.show()
            st.pyplot(fig)

        if(st.sidebar.checkbox("genrewise Sales Comparison")):
            st.subheader("Genrewise Regions Sales Comparison Analysis")
            df1 = df[['Genre', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]
            df1 = df1.groupby("Genre").sum()
            fig = plt.figure()
            sns.heatmap(df1, annot=True,fmt=".2f")
            plt.show()
            st.pyplot(fig)

        if(st.sidebar.checkbox("Global sales data")):
            fig = plt.figure()
            plt.hist(df["Global_Sales"],bins=20)
            plt.xticks(np.arange(0,80,5))
            plt.xlabel("Sale in million")
            plt.show()
            st.pyplot(fig)

        if(st.sidebar.checkbox("Yearly analysis")):
            fig = plt.figure()
            plt.hist(df["Year"],bins=40)
            plt.xticks(rotation=90)
            plt.ylabel("Games released")
            plt.show()
            st.pyplot(fig)

        if(st.sidebar.checkbox("Region wise Global Sales")):
            st.subheader("Region wise Global Sales analysis")
            chart_type=st.sidebar.selectbox("Select chart type: ",["Pie Chart","Bar Graph"])
            if(chart_type=="Pie Chart"):
                fig = plt.figure()
                df2= df[["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]]
                df2= df2.sum().reset_index()
                plt.pie(df2[0],labels=df2["index"], autopct='%1.2f%%')
                plt.show()
                st.pyplot(fig)
            else:
                fig = plt.figure()
                df2= df[["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]]
                df2= df2.sum().reset_index()
                plt.bar(df2["index"],df2[0])
                plt.ylabel("Global Sales")
                plt.xlabel("Regions")
                plt.show()
                st.pyplot(fig)

        if(st.sidebar.checkbox("Genre wise Global Sales")):
            st.subheader("Genre wise Global Sales analysis")
            chart_type=st.sidebar.selectbox("Select chart type: ",["Pie Chart","Bar Graph"])
            if(chart_type=="Pie Chart"):
                fig = plt.figure()
                x_val=df["Genre"].unique()
                y_val=df.groupby("Genre")["Global_Sales"].sum()
                plt.pie(y_val,labels=x_val, autopct='%1.2f%%')
                plt.show()
                st.pyplot(fig)
            else:
                fig = plt.figure()
                x_val=df["Genre"].unique()
                y_val=df.groupby("Genre")["Global_Sales"].sum()
                plt.bar(x_val,y_val,color="maroon")
                plt.xticks(rotation=90)
                plt.xlabel("Genre's type")
                plt.ylabel("Global Sale")
                plt.show()
                st.pyplot(fig)
                
        if(st.sidebar.checkbox("Year wise Global Sales")):
            st.subheader("Year wise Global Sales analysis")
            fig = plt.figure()
            dfyear = df.groupby(['Year'])['Global_Sales'].sum()
            dfyear = dfyear.reset_index()
            sns.barplot(x="Year", y="Global_Sales", data=dfyear)
            plt.xticks(rotation=90)
            plt.ylabel("Global Sale")
            plt.grid(True)
            plt.show()
            st.pyplot(fig)

        if(st.sidebar.checkbox("Platfrom wise Global Sales")):
            st.subheader("Platform wise Global Sales analysis")
            fig = plt.figure()
            dfyear = df.groupby(['Platform'])['Global_Sales'].sum().reset_index().sort_values("Global_Sales",ascending=False) 
            dfyear = dfyear.reset_index()
            sns.barplot(x="Platform", y="Global_Sales", data=dfyear)
            plt.xticks(rotation=90,fontsize=12)
            plt.xlabel("Platform",fontsize=12)
            plt.ylabel("Global Sale",fontsize=12)
            plt.grid(True)
            plt.show()
            st.pyplot(fig)

        if(st.sidebar.checkbox("Year wise games released")):
            st.subheader("Year wise games released analysis")
            fig = plt.figure()
            yrgame = df.groupby('Year')['Name'].count()
            sns.countplot(x="Year", data=df, order=yrgame.index)
            plt.xticks(rotation=90)
            plt.ylabel("No of games released")
            plt.grid(True)
            plt.show()
            st.pyplot(fig)

        if(st.sidebar.checkbox("Genre wise Games Released")):
            st.subheader("Genre wise Games released analysis")
            fig = plt.figure()
            sns.countplot(x="Genre", data=df, order = df['Genre'].value_counts().index)
            plt.xticks(rotation=90,fontsize=12)
            plt.xlabel("Genre",fontsize=12)
            plt.grid(True)
            plt.show()
            st.pyplot(fig)

        if(st.sidebar.checkbox("Platform wise Games Released")):
            st.subheader("Platform wise Games released analysis")
            fig = plt.figure()
            sns.countplot(x="Platform", data=df, order = df['Platform'].value_counts().index)
            plt.xticks(rotation=90,fontsize=12)
            plt.grid(True)
            plt.show()
            st.pyplot(fig)

                
            
        
            
    if(selectact=="Sales Prediction"):
        st.subheader("Sales Prediction by Supervised Machine learning model")
        na = st.number_input("Enter NA_Sales(in million)",0.1,100.0)
        eu = st.number_input("Enter EU_Sales(in million)",0.1,100.0)
        jp = st.number_input("Enter JP_Sales(in million)",0.1,100.0)
        ot = st.number_input("Enter Other_Sales(in million)",0.1,100.0)

        result=[[na,eu,jp,ot]]
        disp_result={"NA_Sales ":na,
                     "EU_Sales":eu,
                     "JP_Sales":jp,
                     "Other_Sales":ot}
        st.info(result)
        st.json(disp_result)

        st.subheader("Make Prediction")

        regressor = st.sidebar.selectbox('Select ML model',('Linear Regression', 'KNN regressor', 'Decision Tree','Random Forest','SVR'))

        def add_param(regressor):
            params = dict()
            if regressor == 'SVR':
                C = st.sidebar.slider('C', 0.01, 10.0)
                params['C'] = C
                sel_kernel=st.sidebar.radio("Select Kernel",(("linear","rbf")))
                if(sel_kernel=="linear"):
                    params["kernel"]=sel_kernel
                else:
                    params["kernel"]=sel_kernel
                    
            elif regressor == 'Linear Regression':
                st.sidebar.text("No parameters")
            elif regressor == 'KNN regressor':
                K = st.sidebar.slider('K', 1, 15)
                params['K'] = K
            elif regressor == 'Decision Tree':
                max_depth = st.sidebar.slider('max_depth',2,15)
                params['max_depth'] = max_depth
                min_sample = st.sidebar.slider('min_samples_leaf',1,15)
                params['min_samples_leaf'] = min_sample
            else:
                max_depth = st.sidebar.slider('max_depth', 2, 15)
                params['max_depth'] = max_depth
                n_estimators = st.sidebar.slider('n_estimators', 1, 100)
                params['n_estimators'] = n_estimators
            return params
        params = add_param(regressor)

        def get_regressor(regressor, params):
            reg = None
            if regressor == 'SVR':
                reg = SVR(C=params['C'],kernel=params["kernel"])
            elif regressor == 'KNN regressor':
                reg = KNeighborsRegressor(n_neighbors=params['K'])
            elif regressor == 'Decision Tree':
                reg = DecisionTreeRegressor(max_depth=params['max_depth'],min_samples_leaf=params['min_samples_leaf'])
            elif regressor == 'Random Forest':
                reg = RandomForestRegressor(max_depth=params['max_depth'],n_estimators=params['n_estimators'])
            else:
                reg = LinearRegression()
            return reg
        reg = get_regressor(regressor, params)

        
        # Model Creation
        df = pd.read_csv("vgsales.csv")
        # Missing Values:
        if(st.sidebar.checkbox("Any Missing Values")):
            st.text("missing values count")
            st.write(df.isna().sum())
            if(st.button("Remove if present")):
                st.text("Drop missing values rows")
                df.dropna(inplace=True)
                st.write(df.isna().sum())
        x = df.iloc[:,6:-1].values
        y = df.iloc[:,-1].values
        st.write('X data(head()):', pd.DataFrame(x).head())
        st.write('y data:(head())', pd.DataFrame(y).head())
        xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=0)
        reg.fit(xtrain,ytrain)
        ypred = reg.predict(xtest)
        acc = r2_score(ytest, ypred)
        st.write(f'Regression model = {regressor}')
        st.success("Accuracy of model is {}%".format(acc*100))

        prediction = reg.predict(result)
        st.write(f'Global Sales Prediction = ')
        st.success(f"{prediction} million")
        

                
        
if __name__=='__main__':
    main()
    
