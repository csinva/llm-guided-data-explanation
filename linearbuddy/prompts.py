expl_program = '''
{{#system~}}
You are a helpful and terse data-science assistant.
{{~/system}}

{{#user~}}
I am going to fit a linear model on a dataset that predicts {{target}} from the following features:
- {{feature_list_str}}
How do you expect each of these features to affect the prediction?
{{~/user}}

{{#assistant~}}
{{gen 'expl_initial' temperature=0 max_tokens=900}}
{{~/assistant}}
                   
{{#user~}}
Here are the coefficients resulting from fitting the linear model after normalization (higher coefficients yield higher predicted {{target}}):
- {{coef1}}                   
Explain what makes sense and what does not.                
{{~/user}}              
            
{{#assistant~}}
{{gen 'expl_1' temperature=0 max_tokens=900}}
{{~/assistant}}
                   
{{#user~}}
Here are the coefficients resulting from fitting a different linear model after normalization (higher coefficients yield higher predicted {{target}}):
- {{coef2}}                   
Again, explain what makes sense and what does not.                
{{~/user}}                                 
                                      
{{#assistant~}}
{{gen 'expl_2' temperature=0 max_tokens=900}}
{{~/assistant}}
                                      
{{#user~}}
Between the first linear model and the second linear model, which is better and why?
Start your reply with "The first model is better" or "The second model is better" and then explain why.
{{~/user}}                   

{{#assistant~}}
{{gen 'model_comparison' temperature=0 max_tokens=900}}
{{~/assistant}}

'''
