# Using LLM to generate QA bencmarks dataset 🧞

This page shows and explain the steps to create a dataset with questions and answers for evaluation of a RAG-based system that provides up-to-date information about the Electron Ion Collider.

## Step 1: Document Preparation

First, prepare the document that contains all the information about the Electron Ion Collider. This document will be used as the source of truth for generating questions and answers.

## Step 2: Question Generation with ChatGPT

Upload the document to ChatGPT and use it to generate a set of questions. ChatGPT is a powerful language model that can generate meaningful and relevant questions based on the context provided in the document.

## Step 3: Answer Generation with ChatGPT

For each question generated in the previous step, use ChatGPT again to generate the corresponding answer. Ensure that the answers are accurate and relevant to the questions.

## Step 4: Dataset Creation

Combine the questions and answers into a single dataset. This dataset can then be used for evaluation of the RAG-based system.

## Step 5: Evaluation

Use the generated dataset to evaluate the performance of the RAG-based system. This can be done by comparing the system's answers to the answers in the dataset. For each question-answer pair, use the RAG (Retrieval-Augmented Generation) model to calculate the RAGAS score. The RAGAS score is a measure of the relevance of the answer to the question, based on the information in the document.

## Step 6: Iteration and Improvement

Based on the results of the evaluation, make necessary adjustments to the RAG-based system or the question and answer generation process. Repeat the steps above until satisfactory performance is achieved.