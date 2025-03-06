# Guidelines for the Team Ball Action Spotting challenge

Here we present the 3rd [Ball Action Spotting Challenge](). Subscribe (watch) the repo to receive the latest info regarding timeline and prizes!

We propose the SoccerNet challenge to encourage the development of state-of-the-art algorithm for Generic Soccer Video Understanding.
 - **Ball Action Spotting**: Spot the actions related to the ball on a complete video of soccer game among 12 classes. In this third edition, methods are additionally required to identify the team performing the action (left or right).

We provide evaluation servers for the team ball action spotting task. The evaluation server handles predictions for the open [**test**](https://www.codabench.org/competitions/4418/) sets and the segregated [**challenge**](https://www.codabench.org/competitions/4417/) sets.

Winners will tentatively be announced at the CVSports Workshop at CVPR 2025.
Prizes will be revealed soon, stay tuned!


## Who can participate / How to participate?

 - Any individual can participate to the challenge, except the organizers.
 - The participants are recommended to form a team to participate.
 - Each team can have one or more members. 
 - An individual/team can compete on different tasks.
 - An individual associated with multiple teams (for a given task) or a team with multiple accounts will be disqualified.
 - On both tasks, a particpant can only use the video stream as input (visual and/or audio).
 - Participants are allowed to use their own models to extract visual/audio features, provided that these models are trained on public datasets. A reference to the datasets used is required.

## How to win / What is the prize?

 - For each task, the winner is the individual/team who reach the highest performance on the **challenge** set.
 - The metric taken into consideration is the **Team mAP@1 for Team Ball Action Spotting**.
 - The deadline to submit your results is April 24th.
 - In order to be eligible for the prize, we require the individual/team to provide a short report describing the details of the methodology (CVPR format, max 2 pages).

## Important dates

Note that these dates are tentative and subject to changes if necessary.

 - **November 20:** Open evaluation server on the (Open) Test set.
 - **November 20:** Open evaluation server on the (Seggregated) Challenge set.
 - **April 24:** Close evaluation server.
 - **May 1:** Deadline for submitting the report.
 - **TBD:** Tentative full-day workshop at CVPR 2025.

## Clarifications on data usage

**1. On the restriction of private datasets and additional annotations**

SoccerNet is designed to be a research-focused benchmark, where the primary goal is to compare algorithms on equal footing. This ensures that the focus remains on algorithmic innovation rather than data collection or annotation effort. Therefore:
* Any data used for training or evaluation must be publicly accessible to everyone to prevent unfair advantages.
* By prohibiting additional manual annotations (even on publicly available data), we aim to avoid creating disparities based on resources (e.g., time, budget, or manpower). This aligns with our commitment to open-source research and reproducibility.

**2. On cleaning or correcting existing data**

We recognize that publicly available datasets, including SoccerNet datasets, might have imperfections in their labels (around 5% usually). Cleaning or correcting these labels is allowed outside of the challenge period to ensure fairness:
* Participants can propose corrections or improvements to older labels before the challenge officially starts. Such changes will be reviewed and potentially integrated into future versions of SoccerNet. Label corrections can be submitted before or after the challenge for inclusion in future SoccerNet releases, ensuring a fair and consistent dataset during the competition.
* During the challenge, participants should not manually alter or annotate existing labels, as this introduces inconsistency and undermines the benchmark's fairness.
* Fully automated methods for label refinement or augmentation, however, are encouraged. These methods should be described in the technical report to ensure transparency and reproducibility.

**3. Defining “private datasets”**

A dataset is considered “private” if it is not publicly accessible to all participants under the same conditions. For example:
* Older SoccerNet data are not private, as they are available to everyone.
* However, manually modifying or adding annotations (e.g., bounding boxes or corrected labels) to older SoccerNet data during the challenge creates a disparity and would be considered "private" unless those modifications are shared with the community in advance.

**4. Creative use of public data**

We fully support leveraging older publicly available SoccerNet data in creative and automated ways, as long as:
* The process does not involve manual annotations.
* The methodology is clearly described and reproducible.
* For instance, if you develop an algorithm that derives additional features or labels (e.g., bounding boxes) from existing data, this aligns with the challenge's goals and is permitted.

**5. Data sharing timeline:**

To ensure fairness, we decided that any new data must be published or shared with all participants through Discord at least one month before the challenge deadline. This aligns with the CVsports workshop timeline and allows all teams to retrain their methods on equal footing.

For any further doubt or concern, please raise an issue in that repository, or contact us directly on [Discord](https://discord.gg/SM8uHj9mkP).
