# AI Usage Log

This document tracks the prompts provided by the user for Assignment 1: Lexical & Semantic Retrieval on EB-NeRD and MIND, as required by the assignment deliverables.

## Prompts

*(Prompt logging begins here)*
## Prompt: 2026-08-19
> Lets start with the data cleaning for the data available in the zip files. and then we will parse it to a unified schema, keep working with what we already have and if there is any confusion discuss with me first

## Prompt: 2026-08-19
> ok go ahea with the plan

## Prompt: 2026-08-19
> lets do temporal splitting as mentioned

## Prompt: 2026-08-19
> (User clicked Approve on the Temporal Splitting implementation plan)

## Prompt: 2026-08-19
> where can i view the cleaned and split dataset so i can check it? also what about the merged one

## Prompt: 2026-08-19
> have we done this yet: One-command rebuild — a single script (e.g., make data or python build_pipeline.py) that rebuilds everything from raw files

## Prompt: 2026-08-19
> lets execute Lexical Candidate Generation (BM25) step by step. start with building an inverted index over article text. come up with an implementation plan and let me verify it.

## Prompt: 2026-08-19
> (User clicked Approve on the BM25 Inverted Index implementation plan)

## Prompt: 2026-08-19
> what does query mean in this case that we are constructing make things clear

## Prompt: 2026-08-20
> tell me how will the grading work on codabench for example in this project, like how will we be ranked? is it based on what our top results prediction per user show? and is there a concrete "true" value which we will be graded against? the test data?

## Prompt: 2026-08-20
> ok lets move on to the query building phase, what is the plan to implement this from our inverted index? share it to me and also briefly explain the math behind the calculations

## Prompt: 2026-08-20
> k so when the model is calculating the BM25 it has the inview articles but it doesnt know which ones are clicked and uncliked until later right?

## Prompt: 2026-08-20
> have we done Report recall@K (how many ground-truth clicked articles appear in the top-K candidates) for K ∈ {50, 100, 200}? how did we do it

## Prompt: 2026-08-20
> how is the recall currently doing? how much did we test on and what have we found on our scores

## Prompt: 2026-08-20
> now next part we have to implement embedding based retrieval and we do have embeddings for ebNERD but do we have them for MIND? and if we use different embeddings then would the comparison later on be fair? discuss.

## Prompt: 2026-08-20
> okay lets move ahead with the plan of creating our own embeddings for both the languages and we will use them. before we go on give an estimate as to how long it will take to download and then compute for the embeddings. do it per language and also suggest which base models (like BERT and ROBERTa) you propose and justify

## Prompt: 2026-08-21
> does paraphrase-multilingual-MiniLM-L12-v2 hhave similar embeddings for both english and danish. is it the best choice and why

## Prompt: 2026-08-21
> ok lets choose that and first start downloading all dependencies as mentioned

## Prompt: 2026-08-21
> check status and give me a time estimate

## Prompt: 2026-08-21
> so far wehave the embedding models and computed our own embeddings ? so now I think ANN index is the next step right? lets plan it

## Prompt: 2026-08-21
> why is indexing necessary when we already have the vectors> discuss

## Prompt: 2026-08-21
> one final thing before moving on, are the metrics we already calculated upto Q3 already enough for these small datasets or would they need running again? I will obviously do it again for large later but right now tell me for the small

## Prompt: 2026-08-21
> what do we need to do now for step 5 and 6? i dont think we can submit on small dataset anyway. I plan to run the entire small dataset on this pipeline start to finish overnight so i can have updated results tomorrow and then start with big datasets. but before all of that is there anything we should tackle first?

## Prompt: 2026-08-23
> can u create a kaggle ready zip for all necessary files to run this on large datasets and i will not directly upload the datasets to kaggle myself so do add the links to them (mentioned in assignment pdf but check the web yourself for the exact link etc), create this zip for me so i can upload to kaggle then give me the runnning instructions too

## Prompt: 2026-08-23
> what output will i get from kaggle that i can later submit on codabench

## Prompt: 2026-08-23
> i created dataset (this zip by uploading) but now how do i create or addd data from url for those ebnerd and mind large

## Prompt: 2026-08-23
> like this? it sys failed to load

## Prompt: 2026-08-23
> also this is the directory so what path should i put

## Prompt: 2026-08-23
> can we not write a wget to fetch mind large datasets from huggingface it is literally available there

## Prompt: 2026-08-23
> i am saving and running in bg so i will run the entire notebook again technically

## Prompt: 2026-08-24
> please go through the assignment 1 pdf and understand the assignment clearly. then look at my code analyuse it and report the weak spots which can be improved to make the score better on leaderboard. currently for mind i am getting score of around 0.61 and 0.54 for ebnerd. I want to improve this score and see where the shortcomings are. lastly, tell me which method? bm25 or semantic with embeddings did we choose for our predictions right now and why

## Prompt: 2026-08-24
> please fix and work on these things you are talking about and after that update kaggle_bundle.zip so i can run the large datasets computations there and generate predictions. also tell me steps to run the same

## Prompt: 2026-08-24
> give me a summary of what changes you did and add it in a new doc, mention the changes, the things you ran locally and the things i have to now run too

## Prompt: 2026-08-24
> for the small datasets did u update the metrics file and log the scores we are getting? did it perform better after your changes

## Prompt: 2026-08-24
> briefly explain to me how hybrid works, how it retrieves and ranks

## Prompt: 2026-08-24
> (Pasted my Kaggle notebook cells) I am sending you the code cells of my kaggle notebook i was previously running, correct it wherever required according to the new code so i can generate submission files for codabench

## Prompt: 2026-08-24
> any estimate on how long it is expected to run on kaggle?

## Prompt: 2026-08-24
> (Shared a Kaggle log screenshot showing zip extraction + deletion) is this expected?

## Prompt: 2026-08-24
> (Shared a Kaggle log screenshot of the ranker training AUC) found this while tunning, it is still running

## Prompt: 2026-08-24
> (Shared another ranker training screenshot) but is it an improvement for before? i had 0.61 on mind before? this is what i found a little earlier, also why are we sampling 200k entries

## Prompt: 2026-08-24
> (Shared a Kaggle progress-log screenshot) this looks like it will take time based on the speed and whats left

## Prompt: 2026-08-24
> there is a new version of the assignment in A1.pdf, check it once it only has minor changes i think we dont have to make any change for this as they are only clarifying the use of large datasets for submission but still pls check

## Prompt: 2026-08-24
> do i need to make any other change in my kaggle notebook?

## Prompt: 2026-08-24
> (Shared a Kaggle log screenshot) just clarify cause seeing an auc on hybrid for mind large train to be 0.59 seems low, so would we get a better score on test when i submit?

## Prompt: 2026-08-24
> what exactly is our feature store and why are we using it

## Prompt: 2026-08-24
> if train and val samples have the true "clicked articles by the user" how are we using that information to train our approach and make it better, exactly because we are using only the bm25 and semantic which is depending on history right?

## Prompt: 2026-08-24
> what metric are we using for users with zero history to show them the articles and how are we doing it

## Prompt: 2026-08-24
> you said EB-NeRD's test-period articles were invisible to the model, explain what exact data was the model not seeing and where was it supposed to be at

## Prompt: 2026-08-24
> i dont understand explain with an example

## Prompt: 2026-08-24
> the only metric being affected because of this was the recency one?

## Prompt: 2026-08-25
> bro my kaggle is already 2 hours away from finishing the previous run i dont think my quota can handle running again with this fix :(

## Prompt: 2026-08-25
> i am already running the improved speed version only, just this last recency change isnt there

## Prompt: 2026-08-25
> untrack kaggle instructions and such files which are for me personally and dont serve the code, plus if there are any past logs or any useless stuff delete it

