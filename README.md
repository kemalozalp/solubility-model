# solubility-model
Welcome! In this project, I train a solubility model and deploy it as a web app. 
I use data provided by Lowe _et. al._ (2023).

So far, I have trained a very simple and crude classification model around 0.001 M (or mol/L) cutoff value, meaning that compounds with solubility < 0.001 are insoluble.
Model returns '0' for insoluble and '1' for soluble compounds. I used Python for training the model and Streamlit for building the web app.
I was inspired by the tutorial and code from Data Professor and shared the link to his YouTube video in the references section.

## What's next?
The next step is to deploy the app using Flask or Heroku.
Stay tuned!

## References
- Lowe, C.N., Charest, N., Ramsland, C., Chang, D.T., Martin, T.M. and Williams, A.J., 2023. Transparency in modeling through careful application of OECD’s QSAR/QSPR principles via a curated water solubility data set. Chemical Research in Toxicology, 36(3), pp.465-478.
