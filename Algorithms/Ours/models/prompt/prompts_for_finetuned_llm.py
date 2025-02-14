detect_aggregation_query_wo_example_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>Using f_agg() API, return True to detect when a natural language query involves performing aggregation operations (max, min, avg, group by). 
<|eot_id|><|start_header_id|>user<|end_header_id|>
question: {question}
answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

select_row_wise_wo_example_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>Using f_row() API to select relevant rows in the given table and linked passages that support or oppose the question.
<|eot_id|><|start_header_id|>user<|end_header_id|>

/*
{table}
*/

/*
{linked_passages}
*/

question: {question}
answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

select_passages_wo_example_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>Using the f_passage() API, you can return a list of linked passages that contain partial information relevant to the question.
<|eot_id|><|start_header_id|>user<|end_header_id|>

/*
{table_segment}
*/

/*
{linked_passages}
*/

**Important** Please ensure that you select all passages that are even slightly related to the question, regardless of whether they contain similar or duplicate information. Include all relevant passages without omission.

question: {question}
answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""