import streamlit as st
import asyncio
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Dict, Any, Optional
import json

# UQLM imports
from uqlm import BlackBoxUQ, WhiteBoxUQ, LLMPanel, UQEnsemble
from uqlm.judges import LLMJudge
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="🤖 Hallucination Detector",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .confidence-high { color: #28a745; }
    .confidence-medium { color: #ffc107; }
    .confidence-low { color: #dc3545; }
    .stAlert {
        border-radius: 0.5rem;
    }
    .api-key-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class HallucinationDetector:
    def __init__(self, groq_api_key=None, google_api_key=None, openai_api_key=None, hf_api_key=None):
        self.models = {}
        self.model_info = {}
        self.groq_api_key = groq_api_key
        self.google_api_key = google_api_key
        self.openai_api_key = openai_api_key
        self.hf_api_key = hf_api_key
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize different LLM models for various scorers"""
        try:
            # Set API keys if provided
            if self.groq_api_key:
                os.environ["GROQ_API_KEY"] = self.groq_api_key
            if self.google_api_key:
                os.environ["GOOGLE_API_KEY"] = self.google_api_key
            if self.openai_api_key:
                os.environ["OPENAI_API_KEY"] = self.openai_api_key
            if self.hf_api_key:
                os.environ["HUGGINGFACE_API_KEY"] = self.hf_api_key
            
            # Initialize models with different providers
            self.models = {}
            self.model_info = {}  # Store model information for display
            
            # Try to initialize Groq models if API key is available
            if os.getenv("GROQ_API_KEY"):
                groq_models = [
                    ("groq:llama3-8b-8192", "Llama3 8B"),
                    ("groq:llama3-70b-8192", "Llama3 70B"),
                    ("groq:mistral-saba-24b", "Mistral Saba 24B"),
                    ("groq:mixtral-8x7b-32768", "Mixtral 8x7B"),
                    ("groq:gemma2-9b-it", "Gemma2 9B"),
                    ("groq:llama3.1-8b-instant", "Llama3.1 8B Instant"),
                    ("groq:llama3.1-70b-versatile", "Llama3.1 70B Versatile"),
                    ("groq:llama3.1-405b-reasoning", "Llama3.1 405B Reasoning")
                ]
                
                for model_id, display_name in groq_models:
                    try:
                        self.models[model_id] = init_chat_model(model=model_id)
                        self.model_info[model_id] = f"Groq - {display_name}"
                    except Exception as e:
                        st.warning(f"Failed to initialize {display_name}: {e}")
            
            # Try to initialize Gemini models if API key is available
            if os.getenv("GOOGLE_API_KEY"):
                gemini_models = [
                    ("google_genai:gemini-1.5-flash", "Gemini 1.5 Flash"),
                    ("google_genai:gemini-1.5-pro", "Gemini 1.5 Pro"),
                    ("google_genai:gemini-pro", "Gemini Pro"),
                    ("google_genai:gemini-pro-vision", "Gemini Pro Vision")
                ]
                
                for model_id, display_name in gemini_models:
                    try:
                        self.models[model_id] = init_chat_model(model=model_id)
                        self.model_info[model_id] = f"Google - {display_name}"
                    except Exception as e:
                        st.warning(f"Failed to initialize {display_name}: {e}")
            
            # Try to initialize OpenAI models if API key is available
            if os.getenv("OPENAI_API_KEY"):
                openai_models = [
                    ("openai:gpt-4o", "GPT-4o"),
                    ("openai:gpt-4o-mini", "GPT-4o Mini"),
                    ("openai:gpt-4-turbo", "GPT-4 Turbo"),
                    ("openai:gpt-3.5-turbo", "GPT-3.5 Turbo")
                ]
                
                for model_id, display_name in openai_models:
                    try:
                        self.models[model_id] = init_chat_model(model=model_id)
                        self.model_info[model_id] = f"OpenAI - {display_name}"
                    except Exception as e:
                        st.warning(f"Failed to initialize {display_name}: {e}")
            
            # Try to initialize HuggingFace models if API key is available
            if os.getenv("HUGGINGFACE_API_KEY"):
                hf_models = [
                    ("huggingface:meta-llama/Llama-2-7b-chat-hf", "Llama-2 7B Chat"),
                    ("huggingface:meta-llama/Llama-2-13b-chat-hf", "Llama-2 13B Chat"),
                    ("huggingface:meta-llama/Llama-2-70b-chat-hf", "Llama-2 70B Chat"),
                    ("huggingface:microsoft/DialoGPT-medium", "DialoGPT Medium"),
                    ("huggingface:gpt2", "GPT-2")
                ]
                
                for model_id, display_name in hf_models:
                    try:
                        self.models[model_id] = init_chat_model(model=model_id)
                        self.model_info[model_id] = f"HuggingFace - {display_name}"
                    except Exception as e:
                        st.warning(f"Failed to initialize {display_name}: {e}")
            
            if not self.models:
                st.error("No models could be initialized. Please check your API keys.")
                return False
            
            return True
        except Exception as e:
            st.error(f"Error initializing models: {e}")
            return False
    
    def get_confidence_color(self, score: float) -> str:
        """Get color class based on confidence score"""
        if score >= 0.7:
            return "confidence-high"
        elif score >= 0.4:
            return "confidence-medium"
        else:
            return "confidence-low"
    
    def get_confidence_label(self, score: float) -> str:
        """Get confidence label based on score"""
        if score >= 0.7:
            return "🟢 High Confidence"
        elif score >= 0.4:
            return "🟡 Medium Confidence"
        else:
            return "🔴 Low Confidence (Potential Hallucination)"

async def run_blackbox_uq(detector: HallucinationDetector, prompt: str, num_responses: int = 5, selected_model: Optional[str] = None):
    """Run BlackBoxUQ analysis"""
    try:
        with st.spinner("Running BlackBoxUQ analysis..."):
            # Use selected model or default to first available
            if selected_model and selected_model in detector.models:
                llm = detector.models[selected_model]
            elif detector.models:
                llm = list(detector.models.values())[0]
            else:
                st.error("No models available for BlackBoxUQ")
                return None
            
            bbuq = BlackBoxUQ(
                llm=llm,
                scorers=["semantic_negentropy", "exact_match", "cosine_sim"]
            )
            
            results = await bbuq.generate_and_score(
                prompts=[prompt],
                num_responses=num_responses
            )
            
            return results.to_df()
    except Exception as e:
        st.error(f"BlackBoxUQ Error: {e}")
        return None

async def run_llm_panel(detector: HallucinationDetector, prompt: str):
    """Run LLMPanel analysis with multiple judges"""
    try:
        with st.spinner("Running LLMPanel analysis..."):
            # Create judges with different models
            if len(detector.models) >= 2:
                model_keys = list(detector.models.keys())
                # Use models directly as judges (LLMPanel will create LLMJudge instances automatically)
                judges = [
                    detector.models[model_keys[0]],
                    detector.models[model_keys[1]]
                ]
                
                panel = LLMPanel(
                    llm=detector.models[model_keys[0]],  # Use first model as primary
                    judges=judges,
                    scoring_templates=["continuous", "continuous"]
                )
            else:
                st.error("Need at least 2 models for LLMPanel analysis")
                return None
            
            results = await panel.generate_and_score([prompt])
            return results.to_df()
    except Exception as e:
        st.error(f"LLMPanel Error: {e}")
        return None

async def run_uq_ensemble(detector: HallucinationDetector, prompt: str, config: Optional[Dict[str, Any]] = None, selected_model: Optional[str] = None):
    """Run UQEnsemble analysis"""
    try:
        with st.spinner("Running UQEnsemble analysis..."):
            # Build scorers list based on configuration
            scorers = []
            
            if config:
                # Add black-box scorers
                if "selected_blackbox" in config:
                    scorers.extend(config["selected_blackbox"])
                
                # Add white-box scorers
                if "selected_whitebox" in config:
                    scorers.extend(config["selected_whitebox"])
                
                # Add model-based scorers
                if "model_scorers" in config:
                    for model_name in config["model_scorers"]:
                        if model_name in detector.models:
                            scorers.append(detector.models[model_name])
            else:
                # Default scorers (Chen & Mueller, 2023 ensemble)
                scorers = ["noncontradiction", "exact_match"]
            
            # Create UQEnsemble with configuration
            # Use selected model or default to first available
            if selected_model and selected_model in detector.models:
                primary_llm = detector.models[selected_model]
            elif detector.models:
                primary_llm = list(detector.models.values())[0]
            else:
                st.error("No models available for UQEnsemble")
                return None
            
            ensemble_kwargs = {
                "llm": primary_llm,
                "scorers": scorers
            }
            
            if config:
                if "use_best" in config:
                    ensemble_kwargs["use_best"] = config["use_best"]
                if "sampling_temperature" in config:
                    ensemble_kwargs["sampling_temperature"] = config["sampling_temperature"]
                if "use_n_param" in config:
                    ensemble_kwargs["use_n_param"] = config["use_n_param"]
                if "selected_device" in config:
                    ensemble_kwargs["device"] = config["selected_device"]
                if "nli_model_name" in config:
                    ensemble_kwargs["nli_model_name"] = config["nli_model_name"]
                if "max_calls_per_min" in config:
                    ensemble_kwargs["max_calls_per_min"] = config["max_calls_per_min"]
                if "system_prompt" in config:
                    ensemble_kwargs["system_prompt"] = config["system_prompt"]
            
            uqe = UQEnsemble(**ensemble_kwargs)
            
            results = await uqe.generate_and_score([prompt])
            return results.to_df()
    except Exception as e:
        st.error(f"UQEnsemble Error: {e}")
        return None

def display_results_summary(results_dict: Dict[str, pd.DataFrame]):
    """Display summary of all results"""
    st.subheader("📊 Analysis Summary")
    
    # Create metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if "blackbox" in results_dict and results_dict["blackbox"] is not None:
            semantic_score = results_dict["blackbox"]["semantic_negentropy"].iloc[0]
            st.metric(
                "Semantic Negentropy",
                f"{semantic_score:.3f}",
                delta=None
            )
    
    with col2:
        if "panel" in results_dict and results_dict["panel"] is not None:
            if "judge_1" in results_dict["panel"].columns:
                judge_score = results_dict["panel"]["judge_1"].iloc[0]
                st.metric(
                    "Judge 1 Score",
                    f"{judge_score:.3f}",
                    delta=None
                )
    
    with col3:
        if "ensemble" in results_dict and results_dict["ensemble"] is not None:
            ensemble_score = results_dict["ensemble"]["ensemble_score"].iloc[0]
            st.metric(
                "Ensemble Score",
                f"{ensemble_score:.3f}",
                delta=None
            )
    
    with col4:
        # Calculate overall confidence
        scores = []
        if "blackbox" in results_dict and results_dict["blackbox"] is not None:
            scores.append(results_dict["blackbox"]["semantic_negentropy"].iloc[0])
        if "panel" in results_dict and results_dict["panel"] is not None:
            if "judge_1" in results_dict["panel"].columns:
                scores.append(results_dict["panel"]["judge_1"].iloc[0])
        if "ensemble" in results_dict and results_dict["ensemble"] is not None:
            scores.append(results_dict["ensemble"]["ensemble_score"].iloc[0])
        
        if scores:
            overall_score = float(np.mean(scores))
            detector = HallucinationDetector()
            confidence_class = detector.get_confidence_color(overall_score)
            confidence_label = detector.get_confidence_label(overall_score)
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>Overall Confidence</h4>
                <p class="{confidence_class}">{confidence_label}</p>
                <h3>{overall_score:.3f}</h3>
            </div>
            """, unsafe_allow_html=True)

def display_detailed_results(results_dict: Dict[str, pd.DataFrame]):
    """Display detailed results for each method"""
    st.subheader("🔍 Detailed Analysis")
    
    # Create tabs for different methods
    tab1, tab2, tab3 = st.tabs(["BlackBoxUQ", "LLMPanel", "UQEnsemble"])
    
    with tab1:
        if "blackbox" in results_dict and results_dict["blackbox"] is not None:
            st.write("**BlackBoxUQ Results:**")
            st.dataframe(results_dict["blackbox"])
            
            # Show sampled responses
            if "sampled_responses" in results_dict["blackbox"].columns:
                responses = results_dict["blackbox"]["sampled_responses"].iloc[0]
                st.write(f"**Generated Responses ({len(responses)}):**")
                for i, response in enumerate(responses, 1):
                    with st.expander(f"Response {i}"):
                        st.write(response)
        else:
            st.warning("BlackBoxUQ results not available")
    
    with tab2:
        if "panel" in results_dict and results_dict["panel"] is not None:
            st.write("**LLMPanel Results:**")
            st.dataframe(results_dict["panel"])
        else:
            st.warning("LLMPanel results not available")
    
    with tab3:
        if "ensemble" in results_dict and results_dict["ensemble"] is not None:
            st.write("**UQEnsemble Results:**")
            st.dataframe(results_dict["ensemble"])
        else:
            st.warning("UQEnsemble results not available")

def create_visualizations(results_dict: Dict[str, pd.DataFrame]):
    """Create visualizations for the results"""
    st.subheader("📈 Visualizations")
    
    # Prepare data for plotting
    plot_data = []
    
    if "blackbox" in results_dict and results_dict["blackbox"] is not None:
        bb_data = results_dict["blackbox"]
        if "semantic_negentropy" in bb_data.columns:
            plot_data.append({
                "Method": "BlackBoxUQ",
                "Metric": "Semantic Negentropy",
                "Score": bb_data["semantic_negentropy"].iloc[0]
            })
    
    if "panel" in results_dict and results_dict["panel"] is not None:
        panel_data = results_dict["panel"]
        if "judge_1" in panel_data.columns:
            plot_data.append({
                "Method": "LLMPanel",
                "Metric": "Judge 1 Score",
                "Score": panel_data["judge_1"].iloc[0]
            })
    
    if "ensemble" in results_dict and results_dict["ensemble"] is not None:
        ensemble_data = results_dict["ensemble"]
        if "ensemble_score" in ensemble_data.columns:
            plot_data.append({
                "Method": "UQEnsemble",
                "Metric": "Ensemble Score",
                "Score": ensemble_data["ensemble_score"].iloc[0]
            })
    
    if plot_data:
        df_plot = pd.DataFrame(plot_data)
        
        # Create bar chart
        fig = px.bar(
            df_plot,
            x="Method",
            y="Score",
            color="Score",
            title="Confidence Scores by Method",
            color_continuous_scale="RdYlGn"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Create radar chart
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=df_plot["Score"].values,
            theta=df_plot["Method"].values,
            fill='toself',
            name='Confidence Scores'
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=False,
            title="Confidence Radar Chart"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

def main():
    # Main header
    st.markdown('<h1 class="main-header">🤖 Hallucination Detector</h1>', unsafe_allow_html=True)
    st.markdown("### Comprehensive AI Response Analysis Using UQLM")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # API Key Configuration (moved to top)
    st.sidebar.subheader("🔑 API Keys")
    st.sidebar.markdown('<div class="api-key-section">', unsafe_allow_html=True)
    
    # Check if API keys are already set in environment
    groq_key_env = os.getenv("GROQ_API_KEY")
    google_key_env = os.getenv("GOOGLE_API_KEY")
    openai_key_env = os.getenv("OPENAI_API_KEY")
    hf_key_env = os.getenv("HUGGINGFACE_API_KEY")
    
    if groq_key_env:
        st.sidebar.success("✅ Groq API Key found in environment")
        groq_api_key = groq_key_env
    else:
        groq_api_key = st.sidebar.text_input(
            "Groq API Key",
            type="password",
            help="Get your key from https://console.groq.com/",
            placeholder="gsk_..."
        )
    
    if google_key_env:
        st.sidebar.success("✅ Google API Key found in environment")
        google_api_key = google_key_env
    else:
        google_api_key = st.sidebar.text_input(
            "Google API Key",
            type="password",
            help="Get your key from https://makersuite.google.com/app/apikey",
            placeholder="AIza..."
        )
    
    if openai_key_env:
        st.sidebar.success("✅ OpenAI API Key found in environment")
        openai_api_key = openai_key_env
    else:
        openai_api_key = st.sidebar.text_input(
            "OpenAI API Key",
            type="password",
            help="Get your key from https://platform.openai.com/api-keys",
            placeholder="sk-..."
        )
    
    if hf_key_env:
        st.sidebar.success("✅ HuggingFace API Key found in environment")
        hf_api_key = hf_key_env
    else:
        hf_api_key = st.sidebar.text_input(
            "HuggingFace API Key",
            type="password",
            help="Get your key from https://huggingface.co/settings/tokens",
            placeholder="hf_..."
        )
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Initialize detector with API keys
    detector = HallucinationDetector(
        groq_api_key=groq_api_key, 
        google_api_key=google_api_key,
        openai_api_key=openai_api_key,
        hf_api_key=hf_api_key
    )
    
    # Show available models
    if detector.models:
        st.sidebar.success(f"✅ {len(detector.models)} models initialized")
        st.sidebar.write("**Available models:**")
        for model_id, display_name in detector.model_info.items():
            st.sidebar.write(f"• {display_name}")
    else:
        st.sidebar.error("❌ No models available")
        st.sidebar.write("Please provide valid API keys to continue")
    
    st.sidebar.subheader("Select Scorer Type")
    scorer_type = st.sidebar.selectbox("Scorer Type", ["Black-Box Scorer", "LLM-as-a-Judge", "Ensemble Scorer", "White-Box Scorer"])
    
    # Model selection
    st.sidebar.subheader("🤖 Model Selection")
    if detector.models:
        available_models = list(detector.models.keys())
        model_display_names = [detector.model_info.get(model_id, model_id) for model_id in available_models]
        model_options = {f"{display_name} ({model_id})": model_id for model_id, display_name in zip(available_models, model_display_names)}
        
        selected_model_display = st.sidebar.selectbox(
            "Primary Model",
            list(model_options.keys()),
            index=0,
            help="Select the main model for analysis"
        )
        selected_model = model_options[selected_model_display]
    else:
        selected_model = None

    if scorer_type == "Black-Box Scorer":
        cosine = st.sidebar.checkbox("Cosine Similarity", value=False, help="Cosine Similarity is a measure of the semantic similarity between the generated response and the reference response.", label_visibility="visible")
        exact = st.sidebar.checkbox("Exact Match", value=False, help="Exact Match is a measure of the exact match between the generated response and the reference response.", label_visibility="visible")
        noncontradiction = st.sidebar.checkbox("Non-Contradiction", value=False, help="Non-Contradiction is a measure of the non-contradiction between the generated response and the reference response.", label_visibility="visible")
        semantic = st.sidebar.checkbox("Semantic Negentropy", value=False, help="Semantic Negentropy is a measure of the semantic similarity between the generated response and the reference response.", label_visibility="visible")

        num_responses = st.sidebar.slider("Number of Responses", min_value=2, max_value=10, value=5, help="Number of responses to generate for BlackBoxUQ analysis")

        
    if scorer_type == "LLM-as-a-Judge":
        # Judge selection
        st.sidebar.subheader("👨‍⚖️ Judge Configuration")
        
        # Select judges (models that will act as judges)
        if detector.models:
            available_judges = list(detector.models.keys())
            judge_display_names = [detector.model_info.get(model_id, model_id) for model_id in available_judges]
            
            # Create a mapping for display
            judge_options = {f"{display_name} ({model_id})": model_id for model_id, display_name in zip(available_judges, judge_display_names)}
            
            selected_judge_display = st.sidebar.multiselect(
                "Select Judges",
                list(judge_options.keys()),
                default=list(judge_options.keys())[:2] if len(judge_options) >= 2 else list(judge_options.keys()),
                help="Select which models will act as judges to evaluate responses"
            )
            
            selected_judges = [judge_options[display] for display in selected_judge_display]
        else:
            st.sidebar.warning("No models available. Please provide API keys first.")
            selected_judges = []
        
        # Scoring template selection
        scoring_template = st.sidebar.selectbox(
            "Scoring Template",
            ["true_false_uncertain", "true_false", "continuous", "likert"],
            index=2,  # Default to continuous
            help="Scoring template for judges: true_false_uncertain (0/0.5/1), true_false (0/1), continuous (0-1), likert (1-5 scale normalized)"
        )
        
        # System prompt customization
        system_prompt = st.sidebar.text_area(
            "System Prompt",
            value="You are a helpful assistant.",
            help="Custom system prompt for the LLM judges",
            height=80
        )
        
        # Rate limiting
        max_calls_per_min = st.sidebar.number_input(
            "Max Calls per Minute",
            min_value=1,
            max_value=100,
            value=60,
            help="Rate limit to avoid API errors"
        )

    if scorer_type == "Ensemble Scorer":
        st.sidebar.subheader("🎯 Ensemble Configuration")
        
        # Select black-box scorers
        blackbox_scorers = ["semantic_negentropy", "noncontradiction", "exact_match", "bert_score", "bleurt", "cosine_sim"]
        selected_blackbox = st.sidebar.multiselect(
            "Black-box Scorers",
            blackbox_scorers,
            default=["noncontradiction", "exact_match"],
            help="Select black-box scoring methods to include in the ensemble"
        )
        
        # Select white-box scorers
        whitebox_scorers = ["normalized_probability", "min_probability"]
        selected_whitebox = st.sidebar.multiselect(
            "White-box Scorers",
            whitebox_scorers,
            default=[],
            help="Select white-box scoring methods to include in the ensemble"
        )
        
        # Add model-based scorers (LLM-as-a-Judge)
        if detector.models:
            available_models = list(detector.models.keys())
            model_display_names = [detector.model_info.get(model_id, model_id) for model_id in available_models]
            model_options = {f"{display_name} ({model_id})": model_id for model_id, display_name in zip(available_models, model_display_names)}
            
            selected_model_scorers_display = st.sidebar.multiselect(
                "LLM-as-a-Judge Scorers",
                list(model_options.keys()),
                default=[],
                help="Select models to use as LLM-as-a-Judge scorers in the ensemble"
            )
            
            model_scorers = [model_options[display] for display in selected_model_scorers_display]
        else:
            st.sidebar.warning("No models available. Please provide API keys first.")
            model_scorers = []
        
        # Ensemble weights
        st.sidebar.subheader("⚖️ Ensemble Weights")
        use_custom_weights = st.sidebar.checkbox(
            "Use Custom Weights",
            value=False,
            help="Enable to set custom weights for ensemble components"
        )
        
        if use_custom_weights:
            st.sidebar.write("**Weight Configuration:**")
            total_scorers = len(selected_blackbox) + len(selected_whitebox) + len(model_scorers)
            if total_scorers > 0:
                st.sidebar.info(f"Total scorers: {total_scorers}. Weights will be normalized to sum to 1.")
        
        # Advanced options
        st.sidebar.subheader("🔧 Advanced Options")
        
        use_best = st.sidebar.checkbox(
            "Use Best Response",
            value=True,
            help="Swap original response for uncertainty-minimized response based on semantic entropy clusters"
        )
        
        sampling_temperature = st.sidebar.slider(
            "Sampling Temperature",
            min_value=0.1,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Temperature for generating sampled LLM responses"
        )
        
        use_n_param = st.sidebar.checkbox(
            "Use N Parameter",
            value=False,
            help="Use n parameter for BaseChatModel (speeds up generation but not compatible with all models)"
        )
        
        # Device selection for NLI models
        device_options = ["cpu", "cuda", "mps"]
        selected_device = st.sidebar.selectbox(
            "Device for NLI Models",
            device_options,
            index=0,
            help="Device for semantic_negentropy and noncontradiction scorers"
        )
        
        # NLI model selection
        nli_model_name = st.sidebar.selectbox(
            "NLI Model",
            ["microsoft/deberta-large-mnli", "microsoft/deberta-base-mnli", "roberta-large-mnli"],
            index=0,
            help="NLI model for semantic similarity scoring"
        )
        
        # Rate limiting
        max_calls_per_min = st.sidebar.number_input(
            "Max Calls per Minute",
            min_value=1,
            max_value=100,
            value=60,
            help="Rate limit to avoid API errors"
        )
        
        # System prompt
        system_prompt = st.sidebar.text_area(
            "System Prompt",
            value="You are a helpful assistant.",
            help="Custom system prompt for the LLM",
            height=80
        )


    
    # # Analysis options
    # st.sidebar.subheader("📊 Analysis Options")
    # run_blackbox = st.sidebar.checkbox("BlackBoxUQ", value=True)
    # run_panel = st.sidebar.checkbox("LLMPanel", value=True)
    # run_ensemble = st.sidebar.checkbox("UQEnsemble", value=True)
    
    # num_responses = st.sidebar.slider(
    #     "Number of Responses (BlackBoxUQ)",
    #     min_value=3,
    #     max_value=10,
    #     value=5,
    #     help="Number of responses to generate for BlackBoxUQ analysis"
    # )
    
    # Main input area
    st.subheader("📝 Input")
    
    # Text input
    prompt = st.text_area(
        "Enter your prompt/question:",
        placeholder="e.g., What is the capital of France?",
        height=100
    )
    
    # Example prompts
    with st.expander("💡 Example Prompts"):
        st.write("""
        - What is the capital of France?
        - Explain quantum computing
        - Write a Python function to sort a list
        - What are the benefits of machine learning?
        - Describe the process of photosynthesis
        """)
    
    # Collect ensemble configuration if ensemble scorer is selected
    ensemble_config = None
    if scorer_type == "Ensemble Scorer":
        ensemble_config = {
            "selected_blackbox": selected_blackbox if 'selected_blackbox' in locals() else [],
            "selected_whitebox": selected_whitebox if 'selected_whitebox' in locals() else [],
            "model_scorers": model_scorers if 'model_scorers' in locals() else [],
            "use_best": use_best if 'use_best' in locals() else True,
            "sampling_temperature": sampling_temperature if 'sampling_temperature' in locals() else 1.0,
            "use_n_param": use_n_param if 'use_n_param' in locals() else False,
            "selected_device": selected_device if 'selected_device' in locals() else "cpu",
            "nli_model_name": nli_model_name if 'nli_model_name' in locals() else "microsoft/deberta-large-mnli",
            "max_calls_per_min": max_calls_per_min if 'max_calls_per_min' in locals() else 60,
            "system_prompt": system_prompt if 'system_prompt' in locals() else "You are a helpful assistant."
        }
    
    # Analysis button
    if st.button("🚀 Run Analysis", type="primary"):
        if not prompt.strip():
            st.error("Please enter a prompt to analyze.")
            return
        
        if not detector.models:
            st.error("No models available. Please provide valid API keys.")
            return
        
        # Run analyses based on scorer type
        results = {}
        
        if scorer_type == "Black-Box Scorer":
            # Get num_responses from the configuration
            num_responses = num_responses if 'num_responses' in locals() else 5
            results["blackbox"] = asyncio.run(run_blackbox_uq(detector, prompt, num_responses, selected_model))
        
        elif scorer_type == "LLM-as-a-Judge":
            results["panel"] = asyncio.run(run_llm_panel(detector, prompt))
        
        elif scorer_type == "Ensemble Scorer":
            results["ensemble"] = asyncio.run(run_uq_ensemble(detector, prompt, ensemble_config, selected_model))
        
        # Display results
        if results and any(v is not None for v in results.values()):
            display_results_summary(results)
            display_detailed_results(results)
            create_visualizations(results)
        else:
            st.error("No analysis completed successfully. Please check your configuration.")
    
    # Information section
    with st.expander("ℹ️ About This Tool"):
        st.write("""
        **Hallucination Detector** uses multiple uncertainty quantification methods to assess the reliability of AI-generated responses:
        
        ### Methods Used:
        1. **BlackBoxUQ**: Analyzes response consistency using semantic similarity and exact matching
        2. **LLMPanel**: Uses multiple AI judges to evaluate response quality
        3. **UQEnsemble**: Combines multiple scoring methods for comprehensive analysis
        
        ### Confidence Levels:
        - 🟢 **High Confidence (≥0.7)**: Response is likely accurate
        - 🟡 **Medium Confidence (0.4-0.7)**: Response may have some uncertainty
        - 🔴 **Low Confidence (<0.4)**: Response may contain hallucinations
        
        ### Supported Models:
        - Groq (Llama3, Mistral)
        - Google Gemini
        - OpenAI GPT (if configured)
        
        ### API Keys:
        You can provide API keys in the sidebar or set them as environment variables:
        - `GROQ_API_KEY` for Groq models
        - `GOOGLE_API_KEY` for Gemini models
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using [UQLM](https://github.com/uncertainty-toolbox/llm-uncertainty) and Streamlit"
    )

if __name__ == "__main__":
    main()