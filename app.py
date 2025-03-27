import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import plotly.express as px
import plotly.graph_objs as go

def display_project_overview():
    st.title('🇳🇬 Nigerian Retail Market Basket Analysis')
    
    # Project Overview Section
    st.markdown("""
    ## Project Purpose
    This data science project aims to transform how Nigerian small and medium retailers 
    understand and optimize their sales strategies through advanced analytics.

    ### Business Challenge
    - Limited insights into customer purchasing behavior
    - Inefficient inventory and marketing strategies
    - Missed opportunities for sales optimization

    ### Our Solution
    We use Market Basket Analysis to:
    - Identify frequently co-purchased items
    - Generate actionable retail insights
    - Help retailers make data-driven decisions

    ### Key Techniques
    - Apriori Algorithm
    - Association Rule Mining
    - Data Visualization

    ### Technologies Used
    - Python
    - Pandas
    - MLxtend
    - Plotly
    - Streamlit
    """)

    # Motivation Section
    st.markdown("""
    ### Why This Matters
    In the competitive Nigerian retail landscape, understanding customer 
    purchasing patterns can be a game-changer for small and medium businesses.
    """)

    # Start Analysis Button with custom styling
    if st.button('Start Market Basket Analysis', type='primary'):
        st.session_state['page'] = 'analysis'
        st.rerun()

def generate_Nigerian_retail_data():
    product_categories = [
        ['Rice', 'Palm Oil', 'Tomato', 'Onions'],
        ['Bread', 'Eggs', 'Butter', 'Milk'],
        ['Chicken', 'Spices', 'Plantain', 'Tomato'],
        ['Rice', 'Chicken', 'Spices'],
        ['Bread', 'Eggs', 'Milk'],
        ['Palm Oil', 'Tomato', 'Onions', 'Chicken'],
        ['Rice', 'Beans', 'Pepper'],
        ['Bread', 'Butter', 'Eggs'],
        ['Plantain', 'Chicken', 'Spices'],
        ['Rice', 'Palm Oil', 'Chicken']
    ]
    return product_categories

def perform_market_basket_analysis(min_support, min_confidence):
    # Generate and Prepare Data
    transactions = generate_Nigerian_retail_data()
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    # Perform analysis with error handling
    try:
        # Apriori Algorithm with adjusted min_support
        frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
        
        # Check if frequent_itemsets is empty
        if frequent_itemsets.empty:
            st.warning(f"No frequent itemsets found with min_support={min_support}. Trying with lower support...")
            
            # Try with a lower support threshold
            frequent_itemsets = apriori(df, min_support=0.05, use_colnames=True)
            
            # If still empty, return empty DataFrames
            if frequent_itemsets.empty:
                st.error("Unable to find any frequent itemsets. Please check your data.")
                return pd.DataFrame(), pd.DataFrame()
        
        # Convert frozensets to lists for display
        frequent_itemsets['itemsets'] = frequent_itemsets['itemsets'].apply(lambda x: list(x))
        
        # Generate Association Rules
        try:
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
            
            # Convert frozensets to lists for display
            rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x))
            rules['consequents'] = rules['consequents'].apply(lambda x: list(x))
            
            return frequent_itemsets, rules
        
        except ValueError:
            st.warning(f"No association rules found with min_confidence={min_confidence}. Trying with lower confidence...")
            
            # Try with a lower confidence threshold
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
            
            if rules.empty:
                st.error("Unable to generate association rules. Please check your data.")
                return frequent_itemsets, pd.DataFrame()
            
            # Convert frozensets to lists for display
            rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x))
            rules['consequents'] = rules['consequents'].apply(lambda x: list(x))
            
            return frequent_itemsets, rules
    
    except Exception as e:
        st.error(f"An error occurred during market basket analysis: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

def main():
    # Initialize session state for page navigation
    if 'page' not in st.session_state:
        st.session_state['page'] = 'overview'

    # Conditional rendering based on page state
    if st.session_state['page'] == 'overview':
        display_project_overview()
    else:
        st.title('🇳🇬 Nigerian Retail Market Basket Analysis')
        
        # Option to return to overview
        if st.sidebar.button('Back to Project Overview'):
            st.session_state['page'] = 'overview'
            st.rerun()
        
        # Sidebar for parameters
        st.sidebar.header('Analysis Parameters')
        min_support = st.sidebar.slider('Minimum Support', 0.01, 1.0, 0.1)
        min_confidence = st.sidebar.slider('Minimum Confidence', 0.01, 1.0, 0.5)
        
        # Perform Analysis
        frequent_itemsets, rules = perform_market_basket_analysis(min_support, min_confidence)
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(['Frequent Itemsets', 'Association Rules', 'Recommendations'])
        
        with tab1:
            st.subheader('Frequent Itemsets')
            if not frequent_itemsets.empty:
                st.dataframe(frequent_itemsets)
                
                # Visualize Frequent Itemsets
                # Convert itemsets to strings for plotting
                frequent_itemsets['itemsets_str'] = frequent_itemsets['itemsets'].apply(lambda x: ', '.join(x))
                
                fig = px.bar(
                    frequent_itemsets.sort_values('support', ascending=False).head(10), 
                    x='support', 
                    y='itemsets_str', 
                    orientation='h',
                    title='Top 10 Frequent Itemsets'
                )
                st.plotly_chart(fig)
            else:
                st.info("No frequent itemsets found. Try adjusting the support threshold.")
        
        with tab2:
            st.subheader('Association Rules')
            if not rules.empty:
                # Create string representations of antecedents and consequents
                rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(x))
                rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(x))
                
                st.dataframe(rules[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']])
                
                # Scatter plot of rules
                fig = px.scatter(
                    rules, 
                    x='support', 
                    y='confidence', 
                    color='lift',
                    hover_data={
                        'antecedents_str': True, 
                        'consequents_str': True,
                        'support': ':.3f',
                        'confidence': ':.3f',
                        'lift': ':.3f'
                    },
                    title='Association Rules: Support vs Confidence'
                )
                st.plotly_chart(fig)
            else:
                st.info("No association rules found. Try adjusting the confidence threshold.")
        
        with tab3:
            st.subheader('Top Retailer Recommendations')
            if not rules.empty:
                recommendations = [
                    f"When customers buy {', '.join(rule['antecedents'])}, recommend {', '.join(rule['consequents'])} (Lift: {rule['lift']:.2f})"
                    for _, rule in rules.sort_values('lift', ascending=False).head(5).iterrows()
                ]
                
                for rec in recommendations:
                    st.write(rec)
            else:
                st.info("No recommendations available. Adjust analysis parameters.")

if __name__ == '__main__':
    main()