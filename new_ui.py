import pandas as pd
from sklearn import linear_model
import streamlit as st
from itertools import combinations
import itertools
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pmdarima as pm
import numpy as np
import plotly.graph_objects as go
import os
import base64
import matplotlib
from plotly.subplots import make_subplots

import openai
import pandas as pd
import random


indicators=pd.read_csv("list.csv")
column_name = "Variables" # Replace with the name of your column
answer_choices = ["({}) {}".format(chr(i+97), choice) for i, choice in enumerate(indicators[column_name])]

df= pd.read_csv('df.csv')
df1= df[['Period/Year', 'Year','Period','Amount','GSV','ISM_PMI','US_Corn','US_Sugar','Milk']].copy()


ext_vars = ['GSV','ISM_PMI','US_Corn','US_Sugar','Milk']


input_text = '''
Gross Domestic Product (GDP)
Consumer Price Index (CPI)
Producer Price Index (PPI)
Unemployment Rate
Trade Balance
Balance of Payments (BOP)
Consumer Confidence Index (CCI)
Purchasing Managers Index (PMI)
Retail Sales
Industrial Production Index (IPI)
Inflation Rate
Interest Rate
Gross National Product (GNP)
Money Supply
Foreign Direct Investment (FDI)
Exchange Rate
Current Account
Export and Import Data
Housing Starts
Business Inventories
Durable Goods Orders
Personal Income
Personal Consumption Expenditures (PCE)
New Home Sales
'''
def extract_indicators(text,input_text ):
    openai.api_key = ""
    random.seed(42)

    prompt = f"Based on the following text, please recommend the top 5 economic indicators:\n{input_text}\n\nTop 5 indicators for " + str(text)
    #question = "Which of the following are best economic indicators for predicting "+text+ "?"
    #prompt1 = question + "\n" + "\n".join(answer_choices) + "\n"
    #prompt1 = question
    model = "text-davinci-002"
    # Use the OpenAI GPT-3 language model to extract relevant keywords and phrases
    response = openai.Completion.create(
      engine=model,
      prompt=prompt,
      max_tokens=100,
      n=1,
      stop=None,
      temperature=0.5,
      #response_format="text"
    )

    # Extract the top recommended indicators from the response
    #recommended_indicators = response.choices[0].text.strip()
    best_indicators = response.choices[0].text.strip()
    return best_indicators


def plot_corr_with_target(corr_matrix, target_var):
    # Get the correlations with the target variable
    corr_with_target = corr_matrix[target_var]
    input_vars = corr_matrix.columns

    # Create the horizontal bar plot
    fig = go.Figure(go.Bar(
        x=corr_with_target,
        y=input_vars,
        orientation='h',
        marker=dict(
            color=['rgba(255, 0, 0, 0.5)' if x < 0 else 'rgba(0, 255, 0, 0.5)' for x in corr_with_target]
        )
    ))

    # Set the title and axis labels
    fig.update_layout(
        title=f'Correlation with {target_var}',
        xaxis=dict(title='Correlation'),
        yaxis=dict(title='Input Variable')
    )

    # Set the x-axis range
    fig.update_xaxes(range=[-1, 1])

    # Exclude the target variable from the plot
    fig.update_layout(yaxis=dict(categoryorder='array', categoryarray=[x for x in input_vars if x != target_var]))

    # Show the plot
    st.plotly_chart(fig)

def anomaly_detection_summary(df: pd.DataFrame):
    """
    This function generates a summary of the anomaly detection table
    
    Parameters:
    df (pd.DataFrame): The anomaly detection table
    
    Returns:
    None
    """
    st.info("Anomaly Detection Summary")
    st.write("- The following periods/years have been detected as anomalies:")
    
    # Determine the dynamic column names
    cols = list(df.columns)
    cols.remove('Period/Year')
    
    # Loop through each equation column and display the periods detected as anomalies
    for col in cols:
        periods = df[df[col] == 'Yes']['Period/Year'].tolist()
        if periods:
            st.write(f"  - {col}: {', '.join(str(period) for period in periods)}")
    
    # Display a message if no anomalies have been detected
    if all(df[col].isnull().all() or df[col].eq('No').all() for col in cols):
        st.write("  - No anomalies have been detected for any equation combination.")

def variance_summary(variance_df, target_variable):
    summary = ""

    # Check for high R-squared variables
    high_r_squared = []
    for i, row in variance_df.iterrows():
        if row["Variable"] != target_variable:
            if row["R_squared"] >= 0.5:
                high_r_squared.append(row["Variable"])
    if len(high_r_squared) > 0:
        summary += "The following variables have a high R-squared value with the target variable:\n"
        for var in high_r_squared:
            r_squared = variance_df.loc[var, "R_squared"]
            summary += f"- {var} (R-squared: {r_squared:.2f})\n"
        summary += "\n"

    # Check for low R-squared variables
    low_r_squared = []
    for i, row in variance_df.iterrows():
        if row["Variable"] != target_variable:
            if row["R_squared"] < 0.1:
                low_r_squared.append(row["Variable"])
    if len(low_r_squared) > 0:
        summary += "The following variables have a low R-squared value with the target variable:\n"
        for var in low_r_squared:
            r_squared = variance_df.loc[var, "R_squared"]
            summary += f"- {var} (R-squared: {r_squared:.2f})\n"
        summary += "\n"

    # If there are no high or low R-squared variables, add a note to the summary
    if len(high_r_squared) == 0 and len(low_r_squared) == 0:
        summary += "There are no variables with a high or low R-squared value with the target variable.\n"

    # Print the summary
    st.write(summary)

def correlation_summary(corr_matrix, target_variable):
    summary = ""

    # Check for highly correlated variables
    highly_correlated = []
    for col in corr_matrix.columns:
        if col != target_variable:
            if abs(corr_matrix.loc[target_variable, col]) >= 0.7:
                highly_correlated.append(col)
    if len(highly_correlated) > 0:
        summary += "The following variables are highly correlated with the target variable:\n"
        for col in highly_correlated:
            summary += f"- {col} (Correlation coefficient: {corr_matrix.loc[target_variable, col]:.2f})\n"
        summary += "\n"
    
    # Check for negatively correlated variables
    neg_correlated = []
    for col in corr_matrix.columns:
        if col != target_variable:
            if corr_matrix.loc[target_variable, col] < 0:
                neg_correlated.append(col)
    if len(neg_correlated) > 0:
        summary += "The following variables are negatively correlated with the target variable:\n"
        for col in neg_correlated:
            summary += f"- {col} (Correlation coefficient: {corr_matrix.loc[target_variable, col]:.2f})\n"
        summary += "\n"
    
    # Check for medium positively correlated variables
    med_pos_correlated = []
    for col in corr_matrix.columns:
        if col != target_variable:
            if 0.3 <= corr_matrix.loc[target_variable, col] < 0.7:
                med_pos_correlated.append(col)
    if len(med_pos_correlated) > 0:
        summary += "The following variables are moderately positively correlated with the target variable:\n"
        for col in med_pos_correlated:
            summary += f"- {col} (Correlation coefficient: {corr_matrix.loc[target_variable, col]:.2f})\n"
        summary += "\n"
    
    # Check for highly positively correlated variables
    high_pos_correlated = []
    for col in corr_matrix.columns:
        if col != target_variable:
            if corr_matrix.loc[target_variable, col] >= 0.7:
                high_pos_correlated.append(col)
    if len(high_pos_correlated) > 0:
        summary += "The following variables are highly positively correlated with the target variable:\n"
        for col in high_pos_correlated:
            summary += f"- {col} (Correlation coefficient: {corr_matrix.loc[target_variable, col]:.2f})\n"
        summary += "\n"

    # If there are no highly correlated or negatively correlated variables, add a note to the summary
    if len(highly_correlated) == 0 and len(neg_correlated) == 0 and len(med_pos_correlated) == 0 and len(high_pos_correlated) == 0:
        summary += "There are no highly correlated or negatively correlated variables with the target variable.\n"
    
    return summary



def summarize_table(df):
    """
    Summarizes the input table and returns the summary as a string.

    Parameters:
        df (pandas.DataFrame): The input table to be summarized.

    Returns:
        str: A string summarizing the input table.
    """
    # Find the best model based on adjusted R-squared
    best_model = df.loc[df['adjscore'].idxmax()]

    # Find the model with the highest R-squared value
    highest_rscore_model = df.loc[df['rscore'].idxmax()]

    # Generate the summary string
    summary = f"The best model uses the variables {best_model['best_eq']} and has an adjusted R-squared value of {best_model['adjscore']:.4f}. "
    summary += f"The model with the highest R-squared value ({highest_rscore_model['rscore']:.4f}) also uses the variables {highest_rscore_model['best_eq']}. "
    summary += "It is important to note that the table only shows the results for linear regression models with a single outcome variable and a fixed set of predictor variables. Other models and predictor variables not included in the table may be more suitable for predicting the outcome variable in different contexts."

    return summary


def summarize_table1(df):
    
    
    mape_mean = df['Mape'].mean()
    mape_min = df['Mape'].min()
    mape_max = df['Mape'].max()
    
    st.write("- The mean MAPE is {:.2f}%".format(mape_mean))
    st.write("- The minimum MAPE is {:.2f}%".format(mape_min))
    st.write("- The maximum MAPE is {:.2f}%".format(mape_max))
    st.info(f"The best equation is {df.iloc[df['Mape'].idxmin()]['Eq']} with an MAPE of {df['Mape'].min():.2f}%.")

def header(url):
     st.markdown(f'<p style="color:#FF4B7A;font-size:30px;border-radius:2%;">{url}</p>', unsafe_allow_html=True)
st.set_page_config(layout="wide") 
header('Intelligent Variance Analysis')
#st.title(' :#FF4B7A[Intillegent Variance Analysis]')

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

@st.cache(allow_output_mutation=True)
def get_img_with_href(local_img_path, target_url):
    img_format = os.path.splitext(local_img_path)[-1].replace('.', '')
    bin_str = get_base64_of_bin_file(local_img_path)
    html_code = f'''
        <a href="{target_url}">
            <img src="data:image/{img_format};base64,{bin_str}" style='height: 250px;margin-top: -120px;width: 300px;' />
        </a>'''
    return html_code

gif_html = get_img_with_href('lab_low.png', 'https://docs.streamlit.io')
st.sidebar.markdown(gif_html, unsafe_allow_html=True)


level = st.sidebar.selectbox(
    'Select Level',
    ('Raws', 'Packs', 'Manufacturing'))


df= pd.read_csv('df.csv')
df1= df[['Period/Year', 'Year','Period','Amount','GSV','ISM_PMI','US_Corn','US_Sugar','Milk']].copy()


ext_vars = ['ISM_PMI','US_Corn','US_Sugar','Milk']
ext_vars1= ['GSV','ISM_PMI','US_Corn','US_Sugar','Milk']
external = st.sidebar.multiselect(
    'Select External Variables',
    ext_vars,
    ext_vars[:2])


internal = st.sidebar.multiselect(
    'Select Internal Variables',
    ['GSV'],
    'GSV')
font_css = """
<style>
button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
  font-size: 20px;
}
</style>
"""

st.write(font_css, unsafe_allow_html=True)

tab11,tab1, tab2,tab3,tab4 = st.tabs(["🤖 ChatGPT Help","🔍 EDA", "💹 Regression","🧭 Time Series Modelling","🥽 Anomaly Detection"])
with tab11:
    st.title("Economic Indicators Finder")

# Add a text input field for user input

    text_input = st.text_input("Enter your text here:")

    # Process the input and display the recommended indicators
    if st.button("Find Indicators"):
        recommended_indicators = extract_indicators(text_input,answer_choices)
        
        
        # Display the recommended indicators in a table
        #df11 = pd.DataFrame(recommended_indicators)
        #if all(choice in recommended_indicators for choice in correct_choices)
        st.write(recommended_indicators)
        #st.write(prompt1)
        st.info("The above indicators are recommended for your text input. You can select them from the sidebar to perform EDA on them.")



with tab1:
    col1, col2 = st.columns([4, 2])
    #data = np.random.randn(10, 1)
    st.subheader("Data Evaluated for:")
    col11, col22, col33 = st.columns(3)
    
    #st.write("")
    #st.write("")
    col11.metric("Cost Heads", "1")
    col22.metric("External Factors", (len(external)))
    col33.metric("Internal Factors", "1")
    
    #col2.line_chart(data)

    #col1.subheader("A narrow column with the data")
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df['Period/Year'], y=df['GSV'],
                        mode='lines',
                        name='GSV'))
    fig.add_trace(go.Scatter(x=df['Period/Year'], y=df["Amount"],
                        mode='lines', name='Amount'))
    fig.update_layout(
    xaxis_type = 'category',
    title='Amount Against GSV'    
    )
    st.subheader("Level Against Internal Variable")
    st.plotly_chart(fig, use_container_width=True)

    #with col2:





    def generate_charts(variables):
        charts = {}
        for var in variables:
            #fig = go.Figure()
            # Customize the chart as needed
            #fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6], name=var))
            #charts[var] = fig
            fig1 = make_subplots(specs=[[{"secondary_y": True}]])
            fig1.add_trace(
                go.Scatter(x=df['Period/Year'], y=df['Amount/GSV'], name="Amount/GSV"),
                secondary_y=False,)

            fig1.add_trace(
                go.Scatter(x=df['Period/Year'], y=df[var], name=var),
                secondary_y=True,)
            fig1.update_layout(
                            xaxis_type = 'category',
                            title="Amount/GSV Against "+ var 
                                    )

                # Set x-axis title
            fig1.update_xaxes(title_text="Year Period")

            charts[var] = fig1
        return charts



    st.subheader('Level Against External Variables')
    # Generate the charts based on the selected variables
    charts = generate_charts(external)

        # Display the charts
    with st.expander("Charts"):
        cols = st.columns(1)
        for i, (var, fig) in enumerate(charts.items()):
            with cols[i%1]:
                st.plotly_chart(fig, use_container_width=True, height=400)
    
    col3, col4 = st.columns([4, 2])
    with col3:
        external.append('Amount')
        external =external+internal
        corr = df1[external].corr()
        st.subheader('Correlation Analysis', anchor=None)
        st.table(corr.style.background_gradient(cmap='coolwarm'))

        plot_corr_with_target(corr, "Amount")


        st.subheader('Insights', anchor=None)
        st.info(correlation_summary(df1[external].corr() , 'Amount'))
    with col4:
        r2_scores =[]

        for i in external:
            df2= df1[[i, "Amount"]].dropna()
            X = df2[i].values.reshape(-1, 1)
            y = df2['Amount'].values.reshape(-1, 1)
            ols = linear_model.LinearRegression()
            model = ols.fit(X, y)
            response = model.predict(X)
            r2 = model.score(X, y)
            r2_scores.append(r2)

        variance_pd = pd.DataFrame()
        variance_pd['Variable']= external
        variance_pd['R_squared']=r2_scores
        st.subheader('Explainability', anchor=None)
        st.table(variance_pd.sort_values(by='R_squared', ascending=False).style.background_gradient(cmap='coolwarm'))
        #(variance_summary(variance_pd, 'Amount'))


with tab2:
    
    st.header('Modelling', anchor=None)
    st.info('All variables are lagged for t-1, t-2, t-3 levels along with t level to find the best lag level of each variable.')
    st.subheader('Build Regression', anchor=None)
    eq_vars = st.multiselect(
        'Select Regressors',
        ext_vars1,
        ext_vars1[:2])

    test_list =eq_vars

    res = []
    for sub in range(len(test_list)):
        res.extend(combinations(test_list, len(test_list)))


    values =[]
    for e in test_list:
        vs =[]
        vs.append(e)
        for i in range(1,4):
            vs.append(e+'_shift_down'+str(i))
        values.append(vs)

    comb_values1 =pd.DataFrame()
    comb_values1['key']=test_list
    comb_values1['values1']= values

    comb_values =pd.Series(comb_values1.values1.values,index=comb_values1.key).to_dict()

    combinations2 =[]
    for i in res:
        vars = list(i)
        #print(len(vars))
        if len(vars)==2:
            combinations1 = list(itertools.product(comb_values[vars[0]],comb_values[vars[1]]#,comb_values[vars[2]],
                                                #comb_values[vars[3]],#comb_values[vars[4]]
                                                ))
        if len(vars)==3:
            combinations1 = list(itertools.product(comb_values[vars[0]],comb_values[vars[1]],comb_values[vars[2]],
                                                #comb_values[vars[3]],#comb_values[vars[4]]
                                                ))
        if len(vars)==4:
            combinations1 = list(itertools.product(comb_values[vars[0]],comb_values[vars[1]],comb_values[vars[2]],
                                                comb_values[vars[3]],#comb_values[vars[4]]
                                                ))
        if len(vars)==5:
            combinations1 = list(itertools.product(comb_values[vars[0]],comb_values[vars[1]],comb_values[vars[2]],
                                                comb_values[vars[3]],comb_values[vars[4]]
                                                ))

        combinations2.append(combinations1)


    select_columns =list(itertools.chain.from_iterable(comb_values1['values1'].tolist()))

    final_combinations =[]
    for i in combinations2:
    
        combinations = i
        best_eq =[]
        rscore =[]
        adjscore =[]
        shapes =[]
        for c in combinations:
            features = list(c)
            eq ='Amount~' + '+'.join(c)
            model_lin = sm.OLS.from_formula(eq, data=df)
            result_lin = model_lin.fit()
            if ((result_lin.pvalues <1).all()) ==True:
                best_eq.append(c)
                rscore.append(result_lin.rsquared)
                adjscore.append(result_lin.rsquared_adj)
                shapes.append(df[features].dropna().shape[0])

    models =pd.DataFrame()
    models['best_eq']=best_eq
    models['rscore']=rscore
    models['adjscore']=adjscore
    models['shape']=shapes
    if models.shape[0] > 0:
        final_combinations.append(list(models['best_eq']))


    merged = list(itertools.chain(*final_combinations))


    best_eq =[]
    rscore =[]
    adjscore =[]
    shapes =[]
    for c in merged:
        features = list(c)
        eq ='Amount~' + '+'.join(c)
        model_lin = sm.OLS.from_formula(eq, data=df)
        result_lin = model_lin.fit()
        if ((result_lin.pvalues <1).all()) ==True:
            best_eq.append(c)
            rscore.append(result_lin.rsquared)
            adjscore.append(result_lin.rsquared_adj)
            shapes.append(df[features].dropna().shape[0])


    models =pd.DataFrame()


    models =pd.DataFrame()
    models['best_eq']=best_eq
    models['rscore']=rscore
    models['adjscore']=adjscore
    models['shape']=shapes
    models =models.sort_values('rscore',ascending=False).reset_index(drop=True)
    st.subheader('Top 20 Models', anchor=None)
    
    
    st.table(models.head(20))
    st.subheader('Insights', anchor=None)
    st.info(summarize_table(models.head(20)))

with tab3:
    equations = models['best_eq'].unique()[:3]
    st.subheader('Top 3 Models')
    st.table(models.head(3))
    st.subheader("Select Model")
    Models = st.selectbox(
        'Select Time Series Model',
        ('Arima', 'Exponential Smoothing', 'Auto Regression'))
    header ='Select Equation For '+Models+" Modelling:"
    st.subheader(header)
    eq_vars1 = st.multiselect(
        'Select Equations',
        equations,
        equations[:1])

    def flatten(t):
        return [item for sublist in t for item in sublist]


    def mape_vectorized_v2(a, b): 
        mask = a != 0
        return (np.fabs(a - b)/a)[mask].mean()


    def mean_absolute_percentage_error(y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    @st.cache()
    def datasets(rolling_period,num_rows,df1):
        itr = rolling_period
        start =0
        train_sets=[]
        test_sets =[]
        while itr < num_rows:
            train_sets.append(list(df1.iloc[start:itr].index))
            test_sets.append(list(df1.iloc[itr:itr+1].index))
            start +=1
            itr +=1
        return train_sets,test_sets

    @st.cache()
    def rolling_autoarima(trains,tests,df1,e,Models):

        yhat =[]
        conf_int1 =[]
        for tr, ts in zip(trains,tests):
            train =df1.loc[tr[0]:tr[-1]]
            test=df1.loc[ts[0]:ts[0]]
            if Models =='Arima':
                modl = pm.auto_arima(train.set_index('Period/Year')['Amount'],X=train.set_index('Period/Year')[e], start_p=1, start_q=1, start_P=1, start_Q=1,
                            max_p=5, max_q=5, max_P=5, max_Q=5, seasonal=True,
                            stepwise=True, suppress_warnings=True, D=10, max_D=10,
                            error_action='ignore')
            if Models == 'Exponential Smoothing':
                modl = pm.auto_arima(train.set_index('Period/Year')['Amount'],X=train.set_index('Period/Year')[e], start_p=0, start_q=1,start_d =1,max_d=1,
                        max_p=0, max_q=1, seasonal=False,
                        stepwise=True, suppress_warnings=True,
                        error_action='ignore')
            if Models =='Auto Regression':
                modl = pm.auto_arima(train.set_index('Period/Year')['Amount'],X=train.set_index('Period/Year')[e], start_p=1, start_q=0, start_P=0, start_Q=0,
                        max_p=5, max_q=0, max_P=0, max_Q=0, seasonal=False,
                        stepwise=True, suppress_warnings=True,
                        error_action='ignore')
            preds, conf_int = modl.predict(1, return_conf_int=True,X=test.set_index('Period/Year')[e])      
            yhat.append(preds)
            conf_int1.append(conf_int)
            
        return yhat,conf_int1

    tested3=pd.DataFrame()
    tested3= df[['Period/Year','Amount']].iloc[13:]

    tested_an=pd.DataFrame()
    tested_an=df['Period/Year'].iloc[13:].to_frame()

    Mape=[]
    for e in eq_vars1:
        tested4= pd.DataFrame()
        sel_col =['Period/Year'] +list(e)+['Amount']
        df1=df[sel_col]
        df1= df1.dropna().reset_index(drop=True)
        trains,tests =datasets(13,df1.shape[0],df1)
        yhat_arima,conf_int1_arima = rolling_autoarima(trains,tests,df1,list(e),Models)
        tested4['Period/Year']=df1['Period/Year'].iloc[tests[0][0]:(tests[-1][0]+1)]
        tested4[e]=flatten(yhat_arima)
        tested2 = df1.loc[13:]
        tested2['Arima']=flatten(yhat_arima)
        tested3=tested3.merge(tested4,on='Period/Year',how='left')
        tested2['Arima%error']= abs((tested2['Arima']-tested2['Amount'])/tested2['Amount'])*100
        Mape.append(tested2['Arima%error'].mean())
        tested2[e]='No'
        tested2[e][tested2['Arima%error']>tested2['Arima%error'].mean()]='Yes'
        tested_an=tested_an.merge(tested2[['Period/Year',e]],on='Period/Year',how='left')

    Mape_df= pd.DataFrame()
    Mape_df['Eq']=eq_vars1
    Mape_df['Mape']= Mape
    st.table(Mape_df)
    plot_data=df[['Period/Year','Amount']].iloc[:13].append(tested3,ignore_index=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_data['Period/Year'], y=plot_data['Amount'][:21],
                        mode='lines',name='Amount'))
    for e in eq_vars1:
        fig.add_trace(go.Scatter(x=plot_data['Period/Year'], y=plot_data[e],
                        mode='lines',name=str(e)
                            ))


    fig.update_layout(
    xaxis_type = 'category',
    width=600,
    height=600,
    legend=dict(
        orientation = 'h', xanchor = "center", x = 0.5, y= -0.3
    )
    )
    st.subheader('Compare Models', anchor=None)
    st.plotly_chart(fig, use_container_width=True)
    summarize_table1(Mape_df)

with tab4:
    st.subheader('Detected Anomalies', anchor=None)
    green = "#d9f1be"
    red = "#ffd9d9"
    color_map = {'Yes': red, 'No': green, pd.NaT: 'white'}
    df_colored = tested_an.style.applymap(lambda x: 'background-color: %s' % color_map.get(x, 'white'))
    st.table(df_colored)
    #st.table(tested_an)
    anomaly_detection_summary(tested_an)



hide_streamlit_style = """
            <style>
            
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 