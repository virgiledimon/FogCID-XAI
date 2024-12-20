# FogCID-XAI
## Explainable AI approach for impersonation attack detection in fog computing

https://fogcid-xai.streamlit.app

http://fogcid-xai.virgiledimon.com

The growing adoption of IoT devices has transformed sectors like smart homes, businesses, and cities, generating massive amounts of diverse data. Fog computing, introduced by Cisco, extends cloud computing closer to the network edge to handle real-time processing demands, mobility, and localization. However, it brings significant challenges related to security and privacy, with impersonation attacks emerging as a key threat due to limited resources at the edge.

This project aims to detect and explain impersonation attacks in Fog computing environments using machine learning (ML) models. Although several ML techniques can identify attacks, the black-box nature of these models makes it difficult to understand their decisions. To overcome this, we integrate Explainable Artificial Intelligence (XAI) tools to provide transparency and confidence in the detection process. Specifically, frameworks like SHAP (Shapley Additive Explanations) and LIME (Local Interpretable Model-agnostic Explanations) are used to analyze feature importance and model reasoning at both global and local levels.

By applying XAI to Fog computing, the project seeks to ensure accountability in decision-making, helping network administrators and users better understand the underlying logic of the detection model. This improved transparency not only enhances trust in the system but also facilitates better system monitoring, early attack mitigation, and adaptation to evolving threats.

![Web App view 1](images/Screenshot1.png)

![Web App view 2](images/Screenshot2.png)

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
