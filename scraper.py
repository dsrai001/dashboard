import pip
import time
package_namesA=['numpy'] #packages to install
pip.main(['install'] + package_namesA + ['--upgrade'])
package_namesnewB=['panel'] #packages to install
pip.main(['install'] + package_namesnewB + ['--upgrade'])
package_namesnewC=['glob'] #packages to install
pip.main(['install'] + package_namesnewC + ['--upgrade'])
import glob
import panel as pn
pn.extension('tabulator')
import numpy as np
package_names=['networkx'] #packages to install
pip.main(['install'] + package_names + ['--upgrade'])

package_names2=['pandas'] #packages to install
pip.main(['install'] + package_names2 + ['--upgrade'])
import pandas as pd
import networkx as nx

package_names2=['matplotlib'] #packages to install

pip.main(['install'] + package_names2 + ['--upgrade'])

package_names3=['sklearn'] #packages to install

pip.main(['install'] + package_names3 + ['--upgrade'])

package_names4=['scikit-learn'] #packages to install

pip.main(['install'] + package_names4 + ['--upgrade'])

package_names5=['scikit-learn'] #packages to install

pip.main(['install'] + package_names5 + ['--upgrade'])

package_names6=['scikit-learn'] #packages to install

pip.main(['install'] + package_names6 + ['--upgrade'])
package_names7=['openpyxl'] #packages to install

pip.main(['install'] + package_names7 + ['--upgrade'])

import matplotlib.pyplot as plt
import openpyxl
from sklearn.metrics.pairwise import cosine_similarity

import array as arr

import numpy as np

import matplotlib.patches as mpatches

package_namesIDB=['streamlit'] #packages to install
pip.main(['install'] + package_namesIDB + ['--upgrade'])
import streamlit as st 

st.set_page_config(page_title="Media companies dashboard", page_icon = ":bar_chart:", layout = "wide")
package_namesnew=['requests'] #packages to install
pip.main(['install'] + package_namesnew + ['--upgrade'])
package_namesnew2=['ssl'] #packages to install
pip.main(['install'] + package_namesnew2 + ['--upgrade'])
package_namesnew3=['glob'] #packages to install
pip.main(['install'] + package_namesnew3 + ['--upgrade'])
file = 'https://github.com/dsrai001/dashboard/blob/main/Clean4.xlsx'
import requests 
from io import BytesIO
import ssl
requests.packages.urllib3.disable_warnings()
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context
if 'data' not in pn.state.cache.keys():
    for file in glob.glob('./*.xlsx'):
      if '~$' in file:
        continue
      else:
        dd = pd.read_excel(
            file,
            engine='openpyxl'
        )

        print(dd)

    

    pn.state.cache['data'] = dd.copy()

else: 

    dd = pn.state.cache['data']
print(dd)
df = dd.astype({"COMPANY": str,

                "GROWTH": str,

                "DATE": str,

                "MARKET": str,

                "VALUATION": str,
                "FUNDING": str,

                "LOCATION": str})
 

print(df)



#def convert_to_int(value):

   # if value[-1] == 't':

       # return int(float(value[1:-1]) * 10**12)

    #if value[-1] == 'b': #elif

       # return int(float(value[1:-1]) * 10**9)

    #elif value[-1] == 'm':

        #return int(float(value[1:-1]) * 10**6)

   # else:

       # return int(float(value[1:])) if value[1:].isdigit() else 0

   

print(df['VALUATION'])

#df=df.apply(convert_to_int) 
df["VALUATION"] = np.floor(pd.to_numeric(df["VALUATION"], errors='coerce')).astype('Int64')
df["FUNDING"] = np.floor(pd.to_numeric(df["FUNDING"], errors='coerce')).astype('Int64')
df["DATE"] = np.floor(pd.to_numeric(df["DATE"], errors='coerce')).astype('Int64')
#result = [convert_to_int(value) for value in df['VALUATION']]

#q = pd.DataFrame({'VALUATION': result})
q=df
print(q)

 

#v = q['VALUATION']

#v = v/100000000

#v = v.to_numpy()

#v = v.tolist()

#v = arr.array('d', v)

 

#df_split = df.set_index(['COMPANY']).apply(lambda x: x.str.split(' ', expand=True).stack()).reset_index(level=1, drop=True).reset_index()

 

# Get unique words from MARKET and TYPE

#unique_words = pd.Series(df_split['MARKET'].tolist()).unique()

 

# Create a similarity matrix based on terms used in MARKET AND TYPE

#df_market_and_type = pd.get_dummies(df_split, columns=['MARKET'])

 

#df_market_and_type_grouped = df_market_and_type.groupby('COMPANY').sum()

 

#matrix = cosine_similarity(df_market_and_type_grouped)

 

#### The first line uses the 'pd.get_dummies' function to create a new DataFrame 'df_market_and_type' with one-hot encoded columns for the values in the 'MARKET' column. This means that each unique value in 'MARKET' gets its own binary column, where a value of 1 indicates the presence of that market and 0 indicates its absence.

#### The second line groups the new DataFrame by the 'COMPANY' column and sums the values of each group, creating a new DataFrame 'df_market_and_type_grouped'.

#### The third line uses the 'cosine_similarity' function to create a similarity matrix from the 'df_market_and_type_grouped' DataFrame. The 'cosine_similarity' function calculates the cosine similarity between the rows of the input matrix, which in this case is the 'df_market_and_type_grouped' DataFrame. The cosine similarity is a measure of how similar two vectors are, and ranges from -1 to 1, where 1 indicates that two vectors are identical, and -1 indicates that they are completely different.

 

# Draw a networkx graph

#G = nx.Graph()

 

# Add nodes to the graph

#for i, company in enumerate(df['COMPANY']):

   # G.add_node(company)

 

#grey = "#FFFFFF"

#matrix = pd.DataFrame(matrix, index=df_market_and_type_grouped.index, columns=df_market_and_type_grouped.index)

#print(matrix)

#matrix.to_csv('to gephi3.csv', sep = ',')

# Add edges to the graph based on similarity score

#for i, row in enumerate(matrix.index):

    #for j, col in enumerate(matrix.columns):

      #  if i <= j:

        #    continue

        #val = matrix.iloc[i, j]

       # G.add_edge(row, col, weight=val, colour = grey)

        #if val:

          #  G.add_edge(row, col, weight=val, colour = grey)

           

# Set predefined color codes

#color_map = {'tech': 'orange', 'advertising': 'blue', 'ecommerce': 'red', 'marketplace': 'purple', 'artificial Intelligence' : 'pink', 'telecommunications' : 'green', 'publishing':'magenta'}

 

# Set node color based on words(can change to TYPE or MARKET)

#node_colors = []

#for company in df['COMPANY']:

  #  market_words = df.loc[df['COMPANY'] == company, 'MARKET'].values[0].split(' ')

  #  found_word = False

   # for word in market_words:

      #  if word in color_map:

         #   color = color_map[word]

          #  node_colors.append(color)

           # found_word = True

            #break

   # if not found_word:

      #  node_colors.append("white")

 

#node_alpha = []

#for company in df['COMPANY']:

    #alpha = df.loc[df['COMPANY'] == company, 'GROWTH'].values[0].split(' ')

    #node_alpha.append(alpha)

#node_alpha = [item[0] for item in node_alpha]

#node_alpha = [float(a[0]) for a in node_alpha]

 

# Draw the graph

#nx.draw_random(G, node_color=node_colors, node_size=v, with_labels=True, edge_color=grey, alpha = 0.5)

#plt.show()
print("hello")

package_namesIDB=['streamlit'] #packages to install
pip.main(['install'] + package_namesIDB + ['--upgrade'])
import streamlit as st

package_namesIDB2=['plotly-express'] #packages to install
pip.main(['install'] + package_namesIDB2 + ['--upgrade'])
package_namesIDB3=['os'] #packages to install
pip.main(['install'] + package_namesIDB3 + ['--upgrade'])
import os
import plotly.express as px
st.dataframe(df)
maxValueV = int(float(df['VALUATION'].max()))
minValueV = int(float(df['VALUATION'].min()))
maxValueF = int(float(df['FUNDING'].max()))
minValueF = int(float(df['FUNDING'].min()))
maxValueD = int(float(df['DATE'].max()))
minValueD = int(float(df['DATE'].min()))
st.sidebar.header("Filter results here:")
market= st.sidebar.multiselect("Select the market to filter by:", options=df["MARKET"].unique(), default=df["MARKET"].unique())
valuation= st.sidebar.slider("Choose the company valuation range to filter by:", minValueV, maxValueV)
date= st.sidebar.slider("Choose the company launch date range to filter by:", minValueD, maxValueD)
funding= st.sidebar.slider("Choose the company funding range to filter by:", minValueF, maxValueF)
companies= st.selectbox("Choose the company you would like to view:", options=df["COMPANY"].unique())
df_selection= df.query(" MARKET == @market & VALUATION == @valuation & FUNDING == @funding & DATE == @date")
df_selection2=df.query("COMPANY==@companies & VALUATION == @valuation")
###MAINPAGE
st.title(":bar_chart: Company Dashboard - all media")
st.markdown("##")
total_companies = (df_selection2["COMPANY"].value_counts())
average_valuation = (df_selection["VALUATION"].mean(),1)
average_funding = (df_selection["FUNDING"].mean(),1)

left_column, middle_column, right_column = st.columns(3)
with left_column:
    st.subheader("Total number of companies:")
    st.subheader(total_companies)
with middle_column:
    st.subheader("Average company valuation in $:")
    st.subheader(average_valuation)
with right_column:
    st.subheader("Average company funding in $:")
    st.subheader(average_funding)

st.markdown("---")

#charts

Valuation_by_market=df_selection.groupby(by=["MARKET"]).sum()[["VALUATION"]].sort_values(by="VALUATION")


fig_valbymarket = px.bar(
    Valuation_by_market,
    x="VALUATION",
    y=Valuation_by_market.index,
    orientation="h",
    title="<b>Average valuation by market</b>",
    color_discrete_sequence=["#0083B8"] * len(Valuation_by_market),
    template="plotly_white",
)
fig_valbymarket.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=(dict(showgrid=False))
)

Funding_by_market=df_selection.groupby(by=["MARKET"]).sum()[["FUNDING"]].sort_values(by="FUNDING")


fig_funbymarket = px.bar(
    Funding_by_market,
    x="FUNDING",
    y=Funding_by_market.index,
    orientation="h",
    title="<b>Average funding by market</b>",
    color_discrete_sequence=["#0083B8"] * len(Funding_by_market),
    template="plotly_white",
)
fig_funbymarket.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=(dict(showgrid=False))
)



#Val_by_co = df_selection2.groupby(by=["COMPANY"]).sum()[["VALUATION"]]
#FigVal_by_co = px.bar(
    #Val_by_co,
    #x=Val_by_co.index,
    #y="VALUATION",
    #title="<b>Average valuation by company</b>",
    #color_discrete_sequence=["#0083B8"] * len(Val_by_co),
    #template="plotly_white",
#)
#FigVal_by_co.update_layout(
    #xaxis=dict(tickmode="linear"),
    #plot_bgcolor="rgba(0,0,0,0)",
    #yaxis=(dict(showgrid=False)),
#)

left_column, right_column = st.columns(2)
left_column.plotly_chart(fig_valbymarket, use_container_width=True)
right_column.plotly_chart(fig_funbymarket, use_container_width=True)

#st.markdown("##")
#left_column, right_column = st.columns(2)
#left_column.plotly_chart(FigVal_by_co, use_container_width=True)
#remove streamlit
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)


