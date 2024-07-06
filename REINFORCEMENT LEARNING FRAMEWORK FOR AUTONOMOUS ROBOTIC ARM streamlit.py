import streamlit as st
import gymnasium as gym
from stable_baselines3 import PPO
from sb3_contrib import TQC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.evaluation import evaluate_policy
import os

# Function to run the model
def run_model():
    st.write("Training started...")
    env_id = "FetchPush-v2" 
    env = gym.make(env_id)
    env = DummyVecEnv([lambda: env])


    model = TQC(
        policy="MultiInputPolicy",
        env=env,
        buffer_size=1000000,
        learning_rate=0.001,
        gamma=0.98,
        batch_size=512,
        tau=0.005,
        policy_kwargs=dict(net_arch=[512, 512, 512], n_critics=2),
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy='future',
        ),
        verbose=1,
        tensorboard_log="./tqc_tensorboard/",
    )

    # Training the model
    model.learn(total_timesteps=int(2e5))
    st.write("Training completed!")

def run_random_episodes():
    st.subheader('Randomly Running the Model')
    st.write("Running the episodes...")
    env = gym.make('FetchPush-v2', max_episode_steps=100, render_mode="human")
    episodes = 5

    for episode in range(1, episodes+1):
        state = env.reset()
        done = False
        score = 0 

        while not done:
            action = env.action_space.sample()
            observation, reward, terminated, done, info = env.step(action)
            score += reward

        # Write the total score obtained in the episode
        st.write(f'Episode {episode} - Total Score: {score}')

    st.write('Episodes completed.')
    env.close()


# Function for hyperparameter tuning
def hyperparameter_tuning():
    st.write("Hyperparameter tuning functionality is not implemented yet.")

# Function to upload and run pretrained model
def upload_and_run_model():
    st.write("Upload pretrained model below.")
    uploaded_file = st.file_uploader("Choose a file", type=['zip'])


# Streamlit app layout
def main():
    st.title('RSR ROBOTICS AND AI SOLUTIONS')
    option = st.sidebar.selectbox('Select Option', ['Train Model', 'Randomly Run Model', 'Hyperparameter Tuning', 'Upload Pretrained Model'])

    if option == 'Train Model':
        st.subheader('Train Model')
        st.selectbox(label = "Select Model", options = ["Fetch_reach", "Fetch_push", "Fetch_slide", "Fetch_Pick&Place"])

        if st.button("Start Training"):
            run_model()

    elif option == 'Randomly Run Model':
        run_random_episodes()

    elif option == 'Hyperparameter Tuning':
        st.subheader('Hyperparameter Tuning')
        st.write("You can perform hyperparameter tuning here.")
        hyperparameter_tuning()

    elif option == 'Upload Pretrained Model':
        st.subheader('Upload Pretrained Model')
        st.write("You can upload and run a pretrained model here.")
        upload_and_run_model()

if __name__ == "__main__":
    main()

