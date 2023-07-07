# AE_TextSpotter_combine-eval-and-test_submit



modified based on [AE TextSpotter](https://github.com/whai362/AE_TextSpotter).

AE TextSpotter does not implement training and evaluating part in a unified code but need author to run eval script manully.

I write training and evaluating part in a unified code.

The modification is simple: 

​				①take "test_cfg" and "test_pipeline" from test config file into train config file; 

​				   data in config file also need to be added val part; 

​				②implement an ReCTSEvalHook

