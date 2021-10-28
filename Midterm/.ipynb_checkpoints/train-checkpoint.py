{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "d45a1216",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "f930c4ba",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.impute import SimpleImputer\n",
    "from sklearn.preprocessing import MinMaxScaler, OneHotEncoder\n",
    "from sklearn.compose import ColumnTransformer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "f9717994",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.model_selection import GridSearchCV"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "4435870d",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.neighbors import KNeighborsClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "b2d05c38",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pickle"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8709e2ae",
   "metadata": {},
   "source": [
    "## EDA"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "2ae1acd8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>PassengerId</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "             Survived  Pclass     Sex   Age  SibSp  Parch     Fare Embarked\n",
       "PassengerId                                                                \n",
       "1                   0       3    male  22.0      1      0   7.2500        S\n",
       "2                   1       1  female  38.0      1      0  71.2833        C\n",
       "3                   1       3  female  26.0      0      0   7.9250        S\n",
       "4                   1       1  female  35.0      1      0  53.1000        S\n",
       "5                   0       3    male  35.0      0      0   8.0500        S"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.read_csv('data/train.csv', index_col='PassengerId')\n",
    "df.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1ef3377a",
   "metadata": {},
   "source": [
    "### Numeric vs Target"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "53f82061",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "c:\\users\\imamx\\anaconda3\\envs\\midterm\\lib\\site-packages\\seaborn\\distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).\n",
      "  warnings.warn(msg, FutureWarning)\n",
      "c:\\users\\imamx\\anaconda3\\envs\\midterm\\lib\\site-packages\\seaborn\\distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).\n",
      "  warnings.warn(msg, FutureWarning)\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAcYAAAFzCAYAAACkZanvAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjQuMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/MnkTPAAAACXBIWXMAAAsTAAALEwEAmpwYAABW8ElEQVR4nO3dd3xUZdr/8c81k04gIQkJkJBCkw5SRAXsCCiCrii4inXF36pr3aK77uq6+jy2tZd9sLcVu7IIAopSRCmhF4HQQwIphJDe5v79cSZswASSkJkzM7ne+8orM2dOuY7j5us55y5ijEEppZRSFofdBSillFK+RINRKaWUqkODUSmllKpDg1EppZSqQ4NRKaWUqkODUSmllKojyO4CvCEuLs6kpqbaXYZSSikfkZ6enmeM6VDfZ60iGFNTU1m5cqXdZSillPIRIrK7oc/0VqpSSilVhwajUkopVYcGo1JKKVVHq3jGqJRS3lRVVUVmZibl5eV2l9LqhYWFkZSURHBwcKO30WBUSqkWlpmZSdu2bUlNTUVE7C6n1TLGkJ+fT2ZmJmlpaY3eTm+lKqVUCysvLyc2NlZD0WYiQmxsbJOv3DUYlVLKAzQUfUNzvgcNRqWUCkAiwr333nvk/VNPPcVDDz103G2++OILNm3aVO9nW7Zs4ZxzzmHQoEH07t2badOmtVitF110EYcOHTrp/Tz00EM89dRTJ70ffcaolFKeNn16y+6vEaEUGhrKZ599xv33309cXFyjdvvFF18wfvx4+vTp84vP7rjjDu6++24mTpwIwPr165tUck1NDU6ns97PZs+e3aR9eZpeMSqlVAAKCgpi2rRpPPPMM7/4bNeuXZx33nkMGDCA888/nz179rB06VJmzpzJH/7wBwYNGsT27duP2iY7O5ukpKQj7/v37w/AW2+9xe23335k+fjx4/n+++8BiIyM5N5772XgwIH87//+L1dcccWR9b7//nvGjx8PWKOT5eXlcd999/HSSy8dWafuFeCTTz7JsGHDGDBgAA8++OCRdR599FF69uzJyJEj2bJlS3P/cR1Fg1EppQLUbbfdxvvvv09hYeFRy3/3u99x3XXXsW7dOq6++mruuOMOzjzzTCZMmMCTTz7JmjVr6Nat21Hb3H333Zx33nmMGzeOZ555plG3PktKShg+fDhr167lvvvuY9myZZSUlADw4YcfMmXKlKPWnzx5Mh999NGR9x999BGTJ09m3rx5bNu2jeXLl7NmzRrS09NZtGgR6enpzJgxgzVr1jB79mxWrFjRzH9SR9NgVEqpANWuXTuuvfZann/++aOW//jjj/z6178GYOrUqSxZsuSE+7rhhhvYvHkzV1xxBd9//z2nn346FRUVx93G6XRy+eWXA9YV7NixY/nPf/5DdXU1X3311ZHbsrVOPfVUcnJyyMrKYu3atbRv354uXbowb9485s2bx6mnnsrgwYP5+eef2bZtG4sXL+ayyy4jIiKCdu3aMWHChKb842mQPmNU/mf7dvjkEwgKggkToEcPuytSymfdddddDB48mBtuuOGk99W5c2duvPFGbrzxRvr168eGDRsICgrC5XIdWadu14iwsLCjnitOmTKFF198kZiYGIYOHUrbtm1/cYwrrriCTz75hP379zN58mTA6o94//33c8sttxy17rPPPnvS51QfvWJU/uXDD+GUU+C+++D3v4c+feC11+yuSimfFRMTw5VXXsnrr79+ZNmZZ57JjBkzAHj//fcZNWoUAG3btqWoqKje/Xz99ddUVVUBsH//fvLz80lMTCQ1NZU1a9bgcrnYu3cvy5cvb7CWs88+m1WrVvHqq6/+4jZqrcmTJzNjxgw++eSTI88kx4wZwxtvvEFxcTEA+/btIycnh7POOosvvviCsrIyioqK+M9//tPEfzr102BU/mPePLj6ahgxAvbsgd274fzz4eaboc7/6ZVSR7v33nvJy8s78v6FF17gzTffZMCAAbz77rs899xzgHVF9+STT3Lqqaf+ovHNvHnz6NevHwMHDmTMmDE8+eSTdOzYkREjRpCWlkafPn244447GDx4cIN1OJ1Oxo8fz5w5c440vDlW3759KSoqIjExkU6dOgFw4YUX8utf/5ozzjiD/v37M2nSJIqKihg8eDCTJ09m4MCBjBs3jmHDhp3sPyoAxBjTIjvyZUOHDjU6H6Ofq6iAfv2s26crVkBkpLW8uhrGjYMlS2DZMhgwwN46lQI2b95M79697S5DudX3fYhIujFmaH3r6xWj8g8vvggZGfDMM/8NRbCC8r33IDraunKs86xDKaWaQ4NR+b7qavjnP2H0aBg79pefJyTAY4/B8uXWM0illDoJGozK9331FWRnQ51OxL8wdSoMHgz33w/uBgJKKdUcGozK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\n",
      "text/plain": [
       "<Figure size 504x432 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# compare cage range with target\n",
    "plt.figure(figsize=(7,6))\n",
    "sns.distplot(df.Age[df.Survived == 0], bins=[0,5,12,18,40,120], color='r', label='Not Survived')\n",
    "sns.distplot(df.Age[df.Survived == 1], bins=[0,5,12,18,40,120], color='g', label='Survived')\n",
    "plt.legend();"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "0701539c",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "c:\\users\\imamx\\anaconda3\\envs\\midterm\\lib\\site-packages\\seaborn\\distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).\n",
      "  warnings.warn(msg, FutureWarning)\n",
      "c:\\users\\imamx\\anaconda3\\envs\\midterm\\lib\\site-packages\\seaborn\\distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).\n",
      "  warnings.warn(msg, FutureWarning)\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAcYAAAFzCAYAAACkZanvAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjQuMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/MnkTPAAAACXBIWXMAAAsTAAALEwEAmpwYAAA6X0lEQVR4nO3deXyU5b3//9cnM9nITgggiwmCKEEQBFxwo2IVlWJ7BFFb9VhPsVVPrdr2YL/Wetr6q1WPntbaWiwup9UCUksRqFpFxQURgnEBREBZAggESMhCMlmu3x/3JAwhIdtMMgnvp495zMx9X3PP54bIO9d13/d1m3MOERER8cR0dgEiIiLRRMEoIiISQsEoIiISQsEoIiISQsEoIiISQsEoIiISwt/ZBYRLr169XE5OTmeXISIiUSQvL6/QOZfVms90m2DMyclh1apVnV2GiIhEETPb0trPaChVREQkhIJRREQkhIJRREQkRLc5xigi0tGqqqooKCigoqKis0s55iUkJDBgwABiY2PbvS0Fo4hIGxUUFJCSkkJOTg5m1tnlHLOcc+zdu5eCggIGDRrU7u1pKFVEpI0qKirIzMxUKHYyMyMzMzNsPXcFo4hIOygUo0M4/x4UjCIiXZiZceedd9a/f+ihh7j33nuP+pkFCxawdu3aRtetX7+eCRMmMGrUKIYNG8aMGTPCVuull15KUVFRu7dz77338tBDD7W/oCboGKOISLjMmhXe7bUglOLj43nhhRe466676NWrV4s2u2DBAiZPnkxubu4R677//e9z++23c/nllwPw8ccft6rkmpoafD5fo+uWLFnSqm11loj2GM1skpmtN7ONZjazkfXxZjY3uH6FmeUEl+eY2UEzyw8+Ho9knSIiXZXf72fGjBk88sgjR6zbvHkzF1xwASNHjmTixIls3bqVd999l4ULF/KjH/2IUaNGsWnTpsM+s3PnTgYMGFD/fsSIEQA8/fTT3HrrrfXLJ0+ezBtvvAFAcnIyd955J6eeeiq/+tWvmDZtWn27N954g8mTJwPeDGWFhYXMnDmTxx57rL5NaA/wwQcfZNy4cYwcOZKf/exn9W3uu+8+hg4dyjnnnMP69evb+sfVIhELRjPzAY8BlwC5wNVm1vDXkxuB/c65IcAjwK9D1m1yzo0KPr4bqTpFRLq6W265hWeffZbi4uLDlv/nf/4n119/PR999BHf/OY3+f73v8/48eOZMmUKDz74IPn5+QwePPiwz9x+++1ccMEFXHLJJTzyyCMtGvosKyvjjDPO4MMPP2TmzJmsWLGCsrIyAObOnctVV111WPvp06czb968+vfz5s1j+vTpvPLKK2zYsIH333+f/Px88vLyWLZsGXl5ecyZM4f8/HyWLFnCypUr2/gn1TKR7DGeDmx0zn3unAsAc4DLG7S5HHgm+Ho+MNF0JFtEpFVSU1O57rrr+O1vf3vY8uXLl3PNNdcAcO211/L22283u60bbriBdevWMW3aNN544w3OPPNMKisrj/oZn8/HFVdcAXg92EmTJvHiiy9SXV3N4sWL64dl64wePZrdu3ezY8cOPvzwQzIyMhg4cCCvvPIKr7zyCqNHj+a0007j008/ZcOGDbz11lt84xvfoEePHqSmpjJlypTW/PG0WiSDsT+wLeR9QXBZo22cc9VAMZAZXDfIzD4wszfN7NzGvsDMZpjZKjNbtWfPnvBWLyLShfzgBz9g9uzZ9T219ujXrx/f/va3+cc//oHf7+eTTz7B7/dTW1tb3yb00oiEhITDjiteddVVzJs3j6VLlzJ27FhSUlKO+I5p06Yxf/585s6dy/Tp0wHvesS77rqL/Px88vPz2bhxIzfeeGO796e1ovWs1J3A8c650cAdwHNmltqwkXNulnNurHNubFZWq+4q0jXMmtX4Q0SkgZ49e3LllVcye/bs+mXjx49nzpw5ADz77LOce67Xx0hJSaGkpKTR7bz00ktUVVUB8OWXX7J371769+9PTk4O+fn51NbWsm3bNt5///0mazn//PNZvXo1TzzxxBHDqHWmT5/OnDlzmD9/fv0xyYsvvpgnn3yS0tJSALZv387u3bs577zzWLBgAQcPHqSkpIQXX3yxlX86rRPJs1K3AwND3g8ILmusTYGZ+YE0YK9zzgGVAM65PDPbBAwFdF8pEZEm3Hnnnfzud7+rf//oo49yww038OCDD5KVlcVTTz0FeD2673znO/z2t79l/vz5hx1nfOWVV7jttttISEgAvJNh+vbtS58+fRg0aBC5ubkMGzaM0047rck6fD4fkydP5umnn+aZZ55ptM3w4cMpKSmhf//+HHfccQBcdNFFrFu3jrPOOgvwTur5y1/+wmmnncb06dM59dRT6d27N+PGjWvfH1QzzMugCGzYC7rPgIl4AbgSuMY5tyakzS3ACOfcd83sKuDfnHNXmlkWsM85V2NmJwBvBdvta+r7xo4d67rd/Rib6h2G8boiEWm7devWMWzYsM4uQ4Ia+/swszzn3NjWbCdiPUbnXLWZ3Qq8DPiAJ51za8zs58Aq59xCYDbwZzPbCOwD6vrc5wE/N7MqoBb47tFCUUREJFwieoG/c24JsKTBsntCXlcA0xr53N+Av0WyNhERkcZE68k3IiIinULBKCIiEkLBKCIiEkLBKCIiEkLBKCLShd13330MHz6ckSNHMmrUKFasWNHubS5cuJD7778/DNV51yJ2NbrtlIhImMzKC+/MVDPGHP2a5eXLl7No0SJWr15NfHw8hYWFBAKBFm27uroav7/xCJgyZUrE5yONZuoxioh0UTt37qRXr17Ex8cD0KtXL/r161d/eyeAVatWMWHCBMC7vdO1117L2WefzbXXXsuZZ57JmjX1c64wYcIEVq1aVX+LqeLiYrKzs+vnSC0rK2PgwIFUVVWxadMmJk2axJgxYzj33HP59NNPAfjiiy8466yzGDFiBHfffXcH/mmEj4JRRKSLuuiii9i2bRtDhw7l5ptv5s0332z2M2vXruXVV1/lr3/962G3f9q5cyc7d+5k7NhDk8SkpaUxatSo+u0uWrSIiy++mNjYWGbMmMGjjz5KXl4eDz30EDfffDMAt912G9/73vf4+OOP66d662oUjCIiXVRycjJ5eXnMmjWLrKwspk+fztNPP33Uz0yZMoXExEQArrzySubPnw9490ScOnXqEe2nT5/O3LlzAZgzZw7Tp0+ntLSUd999l2nTpjFq1Chuuukmdu7cCcA777zD1VdfDXi3uuqKdIxRRKQL8/l8TJgwgQkTJjBixAieeeaZw24RFXp7KICkpKT61/379yczM5OPPvqIuXPn8vjjjx+x/SlTpvCTn/yEffv2kZeXxwUXXEBZWRnp6enk5+c3WlNXv62ueowiIl3U+vXr2bBhQ/37/Px8srOzycnJIS8vD4C//e3os2tOnz6dBx54gOLiYkaOHHnE+uTkZMaNG8dtt93G5MmT8fl8pKamMmjQIJ5//nnAu4/ihx9+CMDZZ5992K2uuiIFo4hIF1VaWsr1119Pbm4uI0eOZO3atdx777387Gc/47bbbmPs2LGH3UC4MVOnTmXOnDlceeWVTbaZPn06f/nLX+pvKAxe6M2ePZtTTz2V4cOH849//AOA3/zmNzz22GOMGDGC7dsb3mmwa4jYbac6mm47JSIdTbedii7huu2UeowiIiIhFIwiIiIhFIwiIiIhFIwiIu3QXc7T6OrC+fegYBQRaaOEhAT27t2rcOxkzjn27t1LQkJCWLanC/xFRNpowIABFBQUsGfPns4u5ZiXkJDAgAEDwrItBaOISBvFxsYyaNCgzi5DwkxDqSIiIiEUjCIiIiEUjCIiIiEUjCIiIiEUjCIiIiEUjCIiIiEUjCIiIiEUjCIiIiEUjCIiIiEUjCIiIiEUjCIiIiEUjCIiIiEUjCIiIiEUjCIiIiEUjCIiIiEUjCIiIiEUjCIiIiEUjCIiIiEUjCIiIiEUjCIiIiEUjCIiIiEUjCIiIiEUjCIiIiEUjCIiIiEUjCIiIiEUjCIiIiEUjCIiIiEUjCIiIiEUjCIiIiEUjCIiIiEUjCIiIiEUjCIiIiEUjCIiIiEiGoxmNsnM1pvZRjOb2cj6eDObG1y/wsxyGqw/3sxKzeyHkaxTRESkTsSC0cx8wGPAJUAucLWZ5TZodiOw3zk3BHgE+HWD9Q8D/4xUjSIiIg1Fssd4OrDROfe5cy4AzAEub9DmcuCZ4Ov5wEQzMwAz+zrwBbAmgjWKiIgcJpLB2B/YFvK+ILis0TbOuWqgGMg0s2Tgv4D/jmB9IiIiR4jWk2/uBR5xzpUerZGZzTCzVWa2as+ePR1TmYiIdGv+CG57OzAw5P2A4LLG2hSYmR9IA/YCZwBTzewBIB2oNbMK59zvQj/snJsFzAIYO3asi8ROiIjIsSWSwbgSONHMBuEF4FXANQ3aLASuB5YDU4GlzjkHnFvXwMzuBUobhqKIiEgkRCwYnXPVZnYr8DLgA550zq0xs58Dq5xzC4HZwJ/NbCOwDy88RUREOk0ke4w455YASxosuyfkdQUwrZlt3BuR4kRERBoRrSffiIiIdAoFo4iISAgFo4iISAgFo4iISAgFo4iISAgFo4iISAgFo4iISAgFo4iISAgFo4iISAgFo4iISAgFo4iISAgFo4iISAgFo4iISAgFo4iISAgFo4iISAgFo4iISAgFo4iISAgFo4iISAgFo4iISAgFo4iISAgFo4iISAgFo4iISAgFo4iISAgFo4iISAgFo4iISAgFo4iISAgFo4iISAgFo4iISAgFo4iISAgFo4iISAgFo4iISAgFo4iISAgFo4iISAgFo4iISAgFo4iISAgFo4iISAgFo4iISAgFo4iISAgFo4iISAgFo4iISAgFo4iISAgFo4iISAgFo4iISAgFo4iISAgFo4iISAgFo4iISAgFo4iISAgFo4iISAgFo4iISAgFo4iISAgFo4iISAgFo4iISAgFo4iISIiIBqOZTTKz9Wa20cxmNrI+3szmBtevMLOc4PLTzSw/+PjQzL4RyTpFRETqRCwYzcwHPAZcAuQCV5tZboNmNwL7nXNDgEeAXweXfwKMdc6NAiYBfzQzf6RqFRERqRPJHuPpwEbn3OfOuQAwB7i8QZvLgWeCr+cDE83MnHPlzrnq4PIEwEWwThERkXqRDMb+wLaQ9wXBZY22CQZhMZAJYGZnmNka4GPguyFBeezasweWLgWn3xNERCIlak++cc6tcM4NB8YBd5lZQsM2ZjbDzFaZ2ao9e/Z0fJEdbdEimDsX/vnPzq5ERKTbimQwbgcGhrwfEFzWaJvgMcQ0YG9oA+fcOqAUOKXhFzjnZjnnxjrnxmZlZYWx9ChUVQX5+d7rX/yiU0sREenOIhmMK4ETzWyQmcUBVwELG7RZCFwffD0VWOqcc8HP+AHMLBs4GdgcwVqj37p1UFEBgwbB++97r0VEJOwiFozBY4K3Ai8D64B5zrk1ZvZzM5sSbDYbyDSzjcAdQN0lHecAH5pZPvB34GbnXGGkau0Sdu70nsePh9pa2Lixc+sREemmInoJhHNuCbCkwbJ7Ql5XANMa+dyfgT9HsrYup7AQevTweozg9SBPOWJ0WURE2ilqT76RBvbuhV69oE8f7/2nn3ZuPSIi3ZSCsasoLPSCMS4OsrMVjCIiEdKiYDSzF8zsMjNTkHYG52DfPsjM9N6ffDKsX9+5NYmIdFMtDbrfA9cAG8zsfjM7KYI1SUMHDniXa9QF48CBsL3hlS8iIhIOLQpG59yrzrlvAqfhXTbxqpm9a2Y3mFlsJAsUvGFU8IZSAY47DnbtgmpNBiQiEm4tHho1s0zg34H/AD4AfoMXlP+KSGVySEmJ95yW5j336+cNr+7e3Xk1iYh0Uy26XMPM/g6chHcJxdecc8GL6phrZqsiVZwElZZ6z8nJ3nO/ft7zjh2HXouISFi09DrGJ4LXJNYzs3jnXKVzbmwE6pJQdT3GumA87jjveceOzqlHRKQba+lQ6i8bWbY8nIXIUZSUQHy8d6kGHN5jFBGRsDpqj9HM+uLdGirRzEYDFlyVCvSIcG1Sp6zsUG8RvIv8zQ5NEyciImHT3FDqxXgn3AwAHg5ZXgL8JEI1SUMlJYcHo9/vhaN6jCIiYXfUYHTOPQM8Y2ZXOOf+1kE1SUMlJZCaeviyvn3VYxQRiYDmhlK/5Zz7C5BjZnc0XO+ce7iRj0m4lZUdefZpr17e/KkiIhJWzQ2lJgWfk4/aSiKr4VAqeLPgbN7cKeWIiHRnzQ2l/jH4/N8dU44cIRDwHikphy9Xj1FEJCJaOon4A2aWamaxZvaame0xs29FujjhyIv762RmQlGRpoUTEQmzll7HeJFz7gAwGW+u1CHAjyJVlIQoK/OeezS4OiYz05sWbv/+jq9JRKQba2kw1g25XgY875wrjlA90tDBg95zYuLhy+smFNdwqohIWLV0SrhFZvYpcBD4npllARWRK0vq1QVjYz1GUDCKiIRZS287NRMYD4x1zlUBZcDlkSxMgsrLvWcFo4hIh2hpjxHgZLzrGUM/839hrkcaam4ote5ejSIiEhYtve3Un4HBQD5QE1zsUDBGXlPBqB6jiEhEtLTHOBbIdc65SBYjjSgv9+6s4fMdvjw5GWJjFYwiImHW0rNSPwH6RrIQacLBg0f2FsG7u0ZmpoZSRUTCrKU9xl7AWjN7H6isW+icmxKRquSQ8vLGgxE0+42ISAS0NBjvjWQRchTl5UeekVonM1PBKCISZi29XONNvBlvYoOvVwKrI1iX1GlqKBUUjCIiEdDSuVK/A8wH/hhc1B9YEKGaJNTBg033GHv10jFGEZEwa+nJN7cAZwMHAJxzG4DekSpKQhztGGNmJuzb582ZKiIiYdHSYKx0zgXq3gQv8te/xpHmXPNDqdXVcOBAx9YlItKNtTQY3zSznwCJZvZV4HngxciVJQBUVUFt7dHPSgUdZxQRCaOWBuNMYA/wMXATsAS4O1JFSVBFcJ72+PjG19fNfqPjjCIiYdOiyzWcc7VmtgBY4JzbE9mSpF5dMCYkNL5e08KJiITdUXuM5rnXzAqB9cB6M9tjZvd0THnHuMrgXAoKRhGRDtPcUOrteGejjnPO9XTO9QTOAM42s9sjXt2xrrmhVN1hQ0Qk7JoLxmuBq51zX9QtcM59DnwLuC6ShQnND6Wmp3tzpu7b12EliYh0d80FY6xz7ojuSPA4Y2xkSpJ6zQVjTAxkZCgYRUTCqLlgDLRxnYRD3THGpoZSAXr2VDCKiIRRc2elnmpmjV09bkAT3RgJm+Z6jKBgFBEJs6MGo3POd7T1EmHNnZUKXjDq5BsRkbBp6QX+0hkqKsDvB99Rfj+pmy9VRETCQsEYzSorj358ETSUKiISZgrGaFZRcfRhVPCCsagIamo6pCQRke5OwRjNWhqM4IWjiIi0m4IxmlVWtjwYNZwqIhIWCsZoVlHRsmOMoPlSRUTCRMEYzVrSY6ybSFw9RhGRsFAwRrPW9BgVjCIiYaFgjGYtvVwDFIwiImGiYIxWzrXsrNT0dO9ZwSgiEhYKxmhVUQG1tc0Ho8/nhaOCUUQkLBSM0aq01HtuLhjBG07VWakiImGhYIxWJSXec3PHGEHzpYqIhJGCMVrVBWNLe4wKRhGRsGjufoztYmaTgN8APuBPzrn7G6yPB/4PGAPsBaY75zab2VeB+4E4vBsi/8g5tzSStUadYDAuZgPby0sPX5d3+NsL2MeQffs7qDARke4tYj1GM/MBjwGXALnA1WaW26DZjcB+59wQ4BHg18HlhcDXnHMjgOuBP0eqzqgVPMZYldD87y6VqUnqMYqIhEkkh1JPBzY65z53zgWAOcDlDdpcDjwTfD0fmGhm5pz7wDm3I7h8DZAY7F0eO4I9xqq45oOxIi0J9u/3zmIVEZF2iWQw9ge2hbwvCC5rtI1zrhooBjIbtLkCWO2cq2z4BWY2w8xWmdmqPXv2hK3wqFAXjAmxzTatTO3hXfeoO2yIiLRbVJ98Y2bD8YZXb2psvXNulnNurHNubFZWVscWF2l1Q6nxLQjGtGTvhYZTRUTaLZLBuB0YGPJ+QHBZo23MzA+k4Z2Eg5kNAP4OXOec2xTBOqNTXY8xvoVDqaBgFBEJg0gG40rgRDMbZGZxwFXAwgZtFuKdXAMwFVjqnHNmlg4sBmY6596JYI3Rq6SEGl8MtX5fs00rU3t4LxSMIiLtFrFgDB4zvBV4GVgHzHPOrTGzn5vZlGCz2UCmmW0E7gBmBpffCgwB7jGz/OCjd6RqjUolJS3qLULwrFRQMIqIhEFEr2N0zi0BljRYdk/I6wpgWiOf+yXwy0jWFvVKS1t0fBGgsm4oVdPCiYi0W1SffHNMa02PMUVDqSIi4aJgjFYlJS3uMTq/D9LSFIwiImGgYIxWregxApovVUQkTBSM0aq0tEUX99dTMIqIhIWCMVqVlLRoOrh6CkYRkbBQMEarkpLW9xh1VqqISLspGKNVaSkBHWMUEelwCsZoVFkJVVVUt/CsVAAyM3WHDRGRMFAwRqPgPKmt7jHW1sKBAxEqSkTk2KBgjEb1E4i38hgjQGFhBAoSETl2KBijUStuOVWvd3Aq2e52X0oRkQ6mYIxGrbjlVD0Fo4hIWCgYo1F7gnH37ggUJCJy7FAwRqO6odTWXMeYleU9KxhFRNpFwRiN6nqMrZn5JiEBUlIUjCIi7aRgjEZ1wdiaHiN4w6kKRhGRdlEwRqO2XK4BCkYRkTBQMEaj0lLw+6nxt/KvR8EoItJurTiIJR2mpASSk8Gs8fVvLTty2ZgZXjCuWBHZ2kREujn1GKNRSYl3Ik1r9e7tXceo+VJFRNpMwRiN2hOMNTXeZOIiItImCsZo1J5gBB1nFBFpBwVjNGprMOoifxGRdlMwRqOSEkhNbf3n1GMUEWk3BWM0OnBAQ6kiIp1EwRiN2jqUmpnpXeKhO2yIiLSZgjHaONf2YPT7vXBUj1FEpM0UjNGmosK75KItxxhBs9+IiLSTgjHaHDjgPbelxwgKRhGRdtKUcNEmOIE4KSmwrxWfmzXr0OcLCg69nzEjrOWJiHR36jFGm9BgbIvUVCguDl89IiLHGAVjtKkLxrYeY0xP945TVlSErSQRkWOJgjHatPcYY3q691xUFI5qRESOOQrGaNPeodSMDO9ZwSgi0iYKxmjT3mBUj1FEpF0UjNEmHMcYQcEoItJGCsZoU3eMMTm5bZ+Pj4cePXRPRhGRNlIwRpuSEkhKgph2/NWkp6vHKCLSRgrGaNPWeVJDKRhFRNpMwRhtDhxo+/HFOgpGEZE2UzBGm3D1GIuLvcnIRUSkVRSM0SYcwZiR4d2+qu5EHhERaTFNIh5tSkogO7vZZrWulrerNvFSxRpizFhS8Qm/Tv03TvL31SUbIiLtoGCMNgcONNtjrHY1/KZsKZ/V7GawrxfpMT14NbCO0Xt+yX/0OIfzExK4Anhl1ztszpvV6DZmjNFdN0REGqNgjDYtGEp9vmI1n9Xs5puJp3Nu7BDMjMLaUv5Q9ia/L3+TXqkTuAJIKirvmJpFRLoRHWOMNs0E46rAFt4IfMaFcSdzXtyJmBkAvWKSuSPpQlIsnt/bKkrijaTigx1VtYhIt6FgjCaBAFRWNnm5RpWrYX7FarJ9Pfm3hNFHrE+Kieffe4xnV+0B7rzUpx6jiEgbKBijSTMTiL8d2Mh+V87X40fhs8b/6ob5+zIhbiizR1TzZY1uWCwi0loKxmhylGCscjW8VLmGIb4shvn7HnUzk+NHkFBrPDxcwSgi0loKxmhylGBcHvicIneQryWMrD+u2JSUmASu+TKLBUNr2bZnUyQqFRHpthSM0eQot5x6K7CRATEZnOTr06JNXV6RTc9yeGntwnBWKCLS7SkYo0ndTDUNeoxba/axtXYfZ8cNbra3WKcmI52bV0JeyafsLtsd7kpFRLotBWM0aWIo9Z3AJvzEcEZsTss3lZnMLSvB72J47YvXwlikiEj3FtFgNLNJZrbezDaa2cxG1seb2dzg+hVmlhNcnmlmr5tZqZn9LpI1RpVGgjHgqnk/sJnTYo8nKSa+xZsqT0mkV0UMk0r68O62dykLlIW7WhGRbiliwWhmPuAx4BIgF7jazHIbNLsR2O+cGwI8Avw6uLwC+Cnww0jVF5UaCcZPqndQToDxcSe0blsxRmnPZG7akEagJsC7294NY6EiIt1XJHuMpwMbnXOfO+cCwBzg8gZtLgeeCb6eD0w0M3POlTnn3sYLyGNHcfDyirS0+kWrq7aSbPEMbeFJN6EO9Ezi9A3lDM4YzLKty6h1teGqVESk24pkMPYHtoW8Lwgua7SNc64aKAYyI1hTdCsqgqQk8HtT2Fa5Gj6q2s4o/8AmL+g/mpLMZFK3F3J+9vnsLtvN+sL1YS5YRKT76dIn35jZDDNbZWar9uzZ09nltF9R0aFbRgFrq3dSSTVjYo9v0+ZKMpOJLynnzJSTSYpNYtmWZeGpU0SkG4tkMG4HBoa8HxBc1mgbM/MDacDeln6Bc26Wc26sc25sVlZWO8uNAsXFhwXj6qqt9LA4TvK3fhgVoKRnMgA9dxYzfuB48nflU1RRFIZCRUS6r0gG40rgRDMbZGZxwFVAw6vNFwLXB19PBZY651wEa4puIT3GQE2AD6sKONU/oE3DqAAHgsGYsqOQ87LPo9bV8s7Wd8JUrIhI9xSxYAweM7wVeBlYB8xzzq0xs5+b2ZRgs9lAppltBO4A6i/pMLPNwMPAv5tZQSNntHY/RUX1J968vfVtDlLFqNgBbd5cSS8vGFML9tA7qTfDeg3jra1vUVNbE45qRUS6pYjeqNg5twRY0mDZPSGvK4BpTXw2J5K1RaXiYjj5ZAAWf7YYPzGc3MyE4UcTSIzjYEYKaVu9mW/Ozz6fx/Me55Pdn4SlXBGR7qhLn3zT7YQMpS7asIih/j4kWGz7Npndh/TNXwIwss9I0uPTdRKOiMhRKBijhXNejzEtjQ17N/DZ3s8Y6W94dUvrFeX0JX3LLgB8MT7OPv5s1uxZwxf7v2j3tkVEuiMFY7QoL4fqakhPZ/GGxQCMiG1/MBZn9yFxfwlxB7wp4c49/lwAZuXNave2RUS6IwVjtCgq8p6DwTis1zB6xSS3f7PZ3qUedb3GjMQMRvYZyewPZhOoCbR7+yIi3Y2CMVoEp4MrS4ln2ZZlXHripeHZbLZ38k7dcUaA87LPY0/5Hl5Y90JYvkNEpDtRMEaLYI/xTdtKoCbAxYMvDstmD/TvRa0vhrRgjxEgNyuXQemDeHzV42H5DhGR7kTBGC2CwfjSwY9J9Cdybva5Ydms8/soHtib9C2HeowxFsNNY27izS1vsm7PurB8j4hId6FgjBbBYHx5/0om5EwgwZ8Qtk0XZ/c5rMcIcMPoG4iNiVWvUUSkAQVjtNi7ly/S4bOSzWEbRq1TlN2HtK27sepDM970TurN1NypPPPhM5RXlYf1+0REujIFY7TYt4+Xh3gvLx4S3mAszu6Lr7qGlJ2Hz8/+3bHfpbiymDmfzAnr94mIdGUKxmixdy8vnxzL8WnHc1LmSWHd9P4TjgMgY9OOw5afe/y55GblajhVRCSEgjFKVO3dw2vHV3Px4Isxs7Bue9+Q/jgzMjcUHLbczPjumO+ycsdK8nbkhfU7RUS6KgVjlFhe9TklcY5JQyaFfdvVifEUD8wi87OCI9Zde+q1JPoT1WsUEQlSMEaJl+O34XPGxEETI7L9fUMG0HPDkcGYnpDO1adczXOfPEdxRXFEvltEpCtRMEaJlzP2cWZFL9IS0iKy/b1DB5BWsIfYsooj1t087mbKq8rVaxQRQcEYFXaX7SYvs5KLa0+I2HfsO9G74XHPjduPWDem3xguGnwR/7P8fygLlEWsBhGRrkDBGAVe2fASABf3GBGx79g7NBiMjQynAvzs/J+xp3yPeo0icsxTMEaBxWsXkFUGYzOGR+w7Svv2pDKlxxFnptYZP3A8EwdN5MF3H1SvUUSOaQrGTlZdW81Lm1/j0g0Q0ysrcl9kxt4T+zfZYwT4+Vd+zq6yXTz07kORq0NEJMopGDvZ8m3LKao6wGWfAT17RvS79p04gMwN27Ga2kbXjx84nqm5U3ng3QfYUbKj0TYiIt2dgrGTLd6wGD8+LtoE9OkT0e/aPTyH2IOVZHzedOjdP/F+qmur+clrP4loLSIi0UrB2MkWb1jMOXGDSask4sG4a+RgAPp8uKnJNoN7DuYHZ/yAZz58hjc2vxHRekREopGCsRNtKdrCJ7s/YXIgx1uQFcFjjEBJ/16UZ6bS9yjBCPCzCT/jhIwT+M6L3+Fg1cGI1iQiEm0UjJ1o8YbFAFy2NxMyMiAuLrJfaMaukYPp89HRg7FHbA+e+NoTbNy3kbuX3h3ZmkREooyCsRMt3rCYEzJO4KSdgYgPo9b58tTBpG4vhC+/PGq7CwZdwPfGfo+H33uYRZ8t6pDaRESigYKxk5RXlbP0i6VcduJl2K7dHRaMu0YGZ9d5991m2z588cOM6juK6/5+HVuKtkS4MhGR6KBg7CSvf/E6FdUVXHbiZbBrV4cFY+HJx1Md54d33mm2bYI/geenPU+Nq2HyXydrknEROSYoGDvJ4g2LSYpN4vyc871g7N27Q763Ni6WwmHZsGxZi9oP6TmEF658gU8LP+WKeVdQWV0Z4QpFRDqXgrETOOdY9NkiLjzhQhKqgeLiDusxAhScMQzy8qCwsEXtJ54wkdlTZvPaF69xxbwrqKg+8g4dIiLdhb+zCzgWvb/9fbYd2MYvvvIL2L3bW9iBwbht/CmMnbWI1/44k02TTm+2/YwxM7ju1OuoqK7gpkU38fU5X+f5ac+TEp/SAdWKiHQs9Rg7wfNrnyc2JpbLT74ctm3zFg4c2GHfXzgsm4q0JI5/5+NWfW7GmBn86Wt/4tXPX+Wcp87RCTki0i2px9jBnHM8v/Z5vjr4q6QnpEc+GN868liiA7acdyo5r39ATKCK2rjYFm/uxtNuZGDaQK58/kpO/9PpLJi+gLMGnhXGgkVEOpd6jB3s/e3vs7V4K1fmXuktqAvGAQM6tI7PJ55GfOlB+r//aas/e9Hgi1h+43JS4lL4yjNf4ckPnsQ5F4EqRUQ6noKxgx02jApeMKakQFpah9ax/YxhVKb0YMhL77fp88OyhrHiP1Zw9vFnc+PCG7nmhWt0OYeIdAsaSu1A1bXVPPfxc0waMskbRgUoKOjQ44t1amP9bJx0Oif9423eOXAVgdSkJtvOypvV5LppudNIiUth3pp5/GvTv1h0zSLOHHBmJEoWEekQ6jF2oFc2vcLO0p38+6h/P7Rw27ZOCUaAT79+Dv5ANSe9uLzN24ixGC498VJ+OP6HAJzz5Dnct+w+amprwlWmiEiHUjB2oKfyn6JXj15MHjr50MJODMa9Jw1k5+ghnPLX17Dq9gXZ4IzB/PS8nzI1dyp3v343F/75Qp21KiJdkoKxg+wt38vC9Qv55ohvEucL3kXj4EFv1pvjj++0uvKvn0TKl/sYuvi9dm8rMTaRv17xV56c8iSrdqxixB9G8NQHT+nEHBHpUhSMHeTZj58lUBM4fBh140bveejQTqkJYNvZp7DrlEGM+eNCfBWBdm/PzLhh9A189N2PGH3caL698Nt8fe7X2VW6KwzViohEnoKxA9S6Wh59/1HO6H8Go/qOOrTis8+8504MRsx47wdTSd5dxNg/LgzbZgdlDOL161/nfy76H17e+DKn/OEUXlj3Qti2LyISKQrGDrDos0Vs3LeR28+8/fAVdcF44okdX9Rby+ofu0p2sO6sIYz4y6sMWL4mbF8RYzHccdYdrL5pNcenHc8V867gur9fR1FFUdi+Q0Qk3BSMHeCR9x5hYOpArsi94vAVn30G/fpBcnLnFBZi+eWnsf+4NCbe9QRpW8I77Jmblct7N77HPefdw3MfP8cpv/d6jzr2KCLRyLrLP05jx451q1at6uwyjpC3I4+xT4zlgQsf4Edn/+jwlePHQ3w8vP56o5+d9b/f6oAKD0neV8o3Hn2NqqQEFv/+dkr69zq0spGp5QA497xWfcfmos38+cM/U1BSwCm9T2HhVQsZlDGoHVWLiDTNzPKcc2Nb8xn1GCPs7tfvpmdiT2aMmXH4Cudg7Vo46aTOKawRpT2Teel/byX+QDlfv/5XZL+R79UZRjnpOfzk3J8wNXcqG/ZuIPf3udzz+j2aNUdEooZmvomgNze/yUsbX+KBCx8gLaHBlG+bNnn3YRzbql9kIm7PKYNY8NR/ceHMWVz8wz+wOzebLeedSlnxbqriY7FahzlHVbyfwv49KW/Dd/hifHz1hK8y9rix5O3M4xfLfsFjKx/jx+N/zK2n30pSXNOz8IiIRJqCMUKcc9z12l30S+nHraffemSDumHfKAtGgOKcvrzwl//HyQveZtgLbzHu8abPVv1ywUesuulr7Dh9WKu/JyMxg3nT5pG3I4+fvv5TZr42k4ffe5ibxtzEjDEzGJDasROri4iAgjFinvzgSZYXLOdPX/sTibGJRzbIy/OOLw4f3vHFtYDz+1g39XzWTT2f2LIKEv61lNjKKpwZtTFGQnmAvpt2k7t6G5Nv/l8+mf4Vlt8+Def3tfq7xvQbw5JvLuHtrW/zq7d/xS+X/ZL73rqPrw39GjeOvpELT7iw8T9DEZEI0Mk3EbD9wHaG/344o/qOYun1S4mxRg7lnnceVFbCihVNbqejT74Bmj6ZpomTb3ynn8Xpv/s7I/76GjtHD+Hl/7n5qBOSt0RheSFvbXmLt7e9TWmglHhfPMOzhvODM3/AxBMm0i+lX7u2LyLHjracfKNgDLNaV8vk5ybzxuY3+Oh7HzGk55AjGx04AJmZcOedcP/9TW6rU4KxjQav3syE55ZT1DuVJd+9gIOXXdzubVbXVvPZ3s/I/zKfD7/8kKLKIu+7MgZzXvZ5nD3wbMb1H0duVi7+GA1+iMiR2hKM+tckzO5eejf/3PhPHrv0scZDEWDpUqiuhksu6djiImjTaTlUJMVz0ZPLmPLoKywePYbSfr2a/+BR+GP85GblkpuVy1WnXMW4fuNYtmUZy7YuY+H6hTyV/xQAif5ERh83mnH9xjG231jG9RvHiZknNt5TFxFphnqMYfTUB0/x7YXfZsZpM3h88uOYWeMN/+M/YN482LsXYmOb3F5X6jHW6b25kElPvE5NUg8WP3YbRSeEb9gz9JKXWlfLxn0bWbl9Jat2rGLljpWs3rmag9UHAUiNT2XMcWPqg3Jc/3Fkp2U3/XciIt2ShlI7MRh/9/7v+M9//icXnnAhi69ZfOgOGg0dOAD9+8PUqfDUU0fdZlcMRoCMHfu57Mm3iamu4Z+//T57hud0yPfW1NbwZemXbC7ezJaiLWwu2kzBgQJqnHdLreS4ZLLTsslOzyYnLYfs9Oz6G0YfcZ2piHQLGkrtBGWBMn78rx/z+1W/5/KTLmfO1DlNhyLAM89AaSl873sdV2QH298vg3/M/jGX3fK/XPa9h3nrJ99i08XjoKneWphm1fHF+Oif2p/+qf05e+DZAFTVVLG9ZDtbirewpch7vLTxJWpdLQDp8elkp2ezu2x3/VBsZo/MVn1vZ5qVN6vFbRX+Ii0T0WA0s0nAbwAf8Cfn3P0N1scD/weMAfYC051zm4Pr7gJuBGqA7zvnXo5kra1V62pZ8OkCZr46kw37NnDHmXfw66/++ugngRQWwr33wvnnw7hxHVZrZygZkMXCJ37IRT96nIl3z+akF9/l42supODMXJyv4479xfpiyUnPISc9B7K9ZYGaANuKt7G5aHN97/Knr/+0/jOD0gfVD8Ge2vdUBmcM5vi044n1NT3s3RrVtdWUVJZQEig57DlQE6DG1VDraql1tdTUHnpdty/+GP9hj/WF6/HF+IixGGJ9sST4Eoj3x5PgTyA2JlZDxyJtELFgNDMf8BjwVaAAWGlmC51za0Oa3Qjsd84NMbOrgF8D080sF7gKGA70A141s6HOufbdZj4MNhdtZv7a+Tyd/zRr9qxhaOZQXrvuNS4YdMHRP3jgAHzjG1BSAo891nTvqRsp753BP578L3Kff4MxTyziktseJZCUQFF2HwLJPYipqcF/MEBs4T7iKquIragCoCIpgZLMJPa9v5N9g/uzb0h/igYdR1VSQvNfWltLQlEpPQqLSdpTTI89RfTI+5D48gAYBOJjKc1IIicjiZGXXUjZoAvAjOnDp7N652pW7lhZ/3h+7fP1m42xGAamDqRPch96JvYkMzGT1PhUYmNi8cX48Mf4qaqporyqnPLqcu+5qpzSQCmbizZTUV1BZXUlFdUVVNVWReqP/DCGEe+LI8GXQGJsIs99/By9k3o3+eiT1IfU+FSFaTejUYXWi9gxRjM7C7jXOXdx8P1dAM65X4W0eTnYZrmZ+YEvgSxgZmjb0HZNfV84jjEerDpIUUURxZXFFFcUs79iP5uLNrNp3yY27t9I/pf5bC7aDMAZ/c/glnG3cPWIq5vuJZaUQH4+vPOOF4Y7dsCcOTBtWovq6arHGIEjhkFjAlVkv/Ux/VatJ23rLvwHK3ExMVQnxhEoL6UqIZaqeO/PMaG0krTCEjJ2lxAbcvPk0j4ZHOifRSA50QtJ5/BXVuE/WEnivhIS9x0gcX8JMTW1R5QTiPdjDmID1YcvT0qgKLsv+084jqIc77msT08CyYnsia9ha+0+9lTso/BgIYXlhZQGSikLlOGL8VFcUUyNq6G6tprqmmrifLH08CXSIyaeHvjpUesjqTqGmpJiUiocaQdrSSutJuNAFRkHAmQcqCIlACmVkBKA+GqIcd6j1u+jJj6W6ow0qpN6UJmcSGWCnxpzVJujllpcIIBVVEJlBVQGqK6upKK6koN+R2kclMRBafCxLxH2pPnZnepjd2It+/2Nh3O8L57ePbLondyHPsl96gMzMzGTlPgUkuOSSYkLPgffJ8Um4Y/xN9qj9cf4j5mzg0P/LXW4sC6rdbUEagJUVFfUPyprKg97X1FdQVmgjLKqssOeV2xfQaAmQGVNJZXVlfWvAzWBw95X1VRhZvjM+0XPF+PDZz4SYxOP+Duvfx+XQkp8CkmxSSTFJdEjtkf966TY4Pvg6zhfXKM/H3U/I5H6hSzajjH2B7aFvC8AzmiqjXOu2syKgczg8vcafLZ/5Er1XPPCNSz4dMERy+N98QzuOZgxx43hjjPv4OIhFzM0swU3Fx42DLZv916PHw/PPw9nnhneoqNVg+OGtcAXcfDFf13dbNt6Z59D6vZCMjbtIOPzHWR8vpPkXftI3riVuIoqXAzUxPqojvVTlhxP4fhTOJiZQnlmGuVZ6ZRlpVHeK42Dn35MTaw3I09MdQ1JReUk7y8jPbUPGV/sJP2LLxmwYh0nLWry9y6cGS7GcGZghs/nh6oqqAkdxKgAShr9fEVqDw72TOWgL4aDyakcTEngYN8EKpLiqfXFUB1j1NY4YiuriKvwes/JBwMkxCcTX1RG/Nb9+CsCuLp/OyoqqYr3U5UQSyAhlkBCDwIJsVQlx1IV56//c6n1Gf4B2cSVVTDGPxDWbYGtWwls20xhbSm7k2BXEuyuf1SyK7mA3UkF7Eo2PkqC3RmxBGoCje5XSxiGL8aHcegfvob/CLZ0XcP1DdfBoWCBQ+ESyWXRLDYmllhfLHG+OOJ98d6zP54esT3ISMjwlvm9dbG+WEb3HU1NrffLXo2roaa2hoPVBykJlFAaKKWksoRdpbvYGNhY/740UBqWPw9/jB/DMDMMIyspi223b2v+gxHQpU++MbMZQF3fv9TM1kfieyqpZG3wv7/xt7Zt5N134ayzWvupXkBh274wWj0b+qaZ/Xu26VWNWbujDfW0kHNQE/I/f1V10209h+/bgXLv0SlWNr2q8RwPqtvfRkOxxT+bDkc1zf55RZtu8f9eVfC/8iOn+290/5awpEPqakzDn5ECCrA72tSLbLhv2a3dQCSDcTswMOT9gOCyxtoUBIdS0/BOwmnJZ3HOzQJaPoDexZjZqtYOAXQl3Xn/uvO+gfavq+vO+xeOfYvk4P9K4EQzG2RmcXgn0zS8TcNC4Prg66nAUueNVSwErjKzeDMbBJwIvB/BWkVERIAI9hiDxwxvBV7Gu1zjSefcGjP7ObDKObcQmA382cw2AvvwwpNgu3nAWqAauCUazkgVEZHuL6LHGJ1zS+DwQWvn3D0hryuARk/RdM7dB9wXyfq6gG47TBzUnfevO+8baP+6uu68f+3et24zJZyIiEg4HBsXGImIiLSQgjEKmdkkM1tvZhvNbGZn19MWZvakme02s09ClvU0s3+Z2Ybgc0ZwuZnZb4P7+5GZndZ5lbeMmQ00s9fNbK2ZrTGz24LLu/w+mlmCmb1vZh8G9+2/g8sHmdmK4D7MDZ5UR/AkubnB5SvMLKdTd6CFzMxnZh+Y2aLg+26zf2a22cw+NrN8M1sVXNblfzbrmFm6mc03s0/NbJ2ZnRXO/VMwRhk7NJXeJUAucLV5U+R1NU8Dkxosmwm85pw7EXgt+B68fT0x+JgB/KGDamyPauBO51wucCZwS/DvqTvsYyVwgXPuVGAUMMnMzsSbsvER59wQYD/elI4QMrUj8EiwXVdwG7Au5H1327+vOOdGhVy60B1+Nuv8BnjJOXcycCre32P49s85p0cUPYCzgJdD3t8F3NXZdbVxX3KAT0LerweOC74+DlgffP1H4OrG2nWVB/APvHmBu9U+Aj2A1XizVhUC/uDy+p9TvDPPzwq+9gfbWWfX3sx+DQj+43kBsAiwbrZ/m4FeDZZ1i59NvOvdv2j4dxDO/VOPMfo0NpVexKfD6yB9nHM7g6+/BPoEX3fpfQ4OrY0GVtBN9jE4zJgP7Ab+BWwCipxzddOThNZ/2NSOQN3UjtHsf4Ef481WCF693Wn/HPCKmeWZN0MYdJOfTWAQsAd4KjgU/iczSyKM+6dglE7hvF/duvwp0WaWDPwN+IFz7kDouq68j865GufcKLye1enAyZ1bUfiY2WRgt3Mur7NriaBznHOn4Q0j3mJmh83q35V/NvF67acBf3DOjQbKODRsCrR//xSM0adF0+F1UbvM7DiA4PPu4PIuuc9mFosXis86514ILu5W++icKwJexxtaTDdv6kY4vP76fbPDp3aMVmcDU8xsMzAHbzj1N3Sf/cM5tz34vBv4O94vN93lZ7MAKHDOrQi+n48XlGHbPwVj9GnJVHpdVegUgNfjHZerW35d8OyxM4HikCGRqGRmhjdz0zrn3MMhq7r8PppZlpmlB18n4h07XYcXkFODzRruW2NTO0Yl59xdzrkBzrkcvP+/ljrnvkk32T8zSzKzlLrXwEXAJ3SDn00A59yXwDYzOym4aCLeLGnh27/OPpCqR6MHly8FPsM7rvP/OrueNu7DX4GdQBXeb3g34h2XeQ3YALwK9Ay2NbwzcTcBHwNjO7v+FuzfOXhDNR8B+cHHpd1hH4GRwAfBffsEuCe4/AS8OYs3As8D8cHlCcH3G4PrT+jsfWjFvk4AFnWn/Qvux4fBx5q6f0O6w89myD6OAlYFf0YXABnh3D/NfCMiIhJCQ6kiIiIhFIwiIiIhFIwiIiIhFIwiIiIhFIwiIiIhInqjYhEJLzOrwTvlvM7XnXObO6kckW5Jl2uIdCFmVuqcS27lZwzv//XaZhuLiIZSRboyM0s2s9fMbHXw/nuXB5fnmHdPz//Du0h/oJn9yMxWBu9J99+dW7lI9NJQqkjXkhi86wV4t96ZBnzDOXfAzHoB75lZ3RSCJwLXO+feM7OLgu9Px5sJZKGZneecW9bB9YtEPQWjSNdy0Hl3vQDqJzL//4J3T6jFu51O3e12tjjn3gu+vij4+CD4PhkvKBWMIg0oGEW6tm8CWcAY51xV8I4RCcF1ZSHtDPiVc+6PHVyfSJejY4wiXVsa3r0Fq8zsK0B2E+1eBr4dvH8kZtbfzHp3VJEiXYl6jCJd27PAi2b2Md7dBj5trJFz7hUzGwYs905SpRT4FofuWSciQbpcQ0REJISGUkVEREIoGEVEREIoGEVEREIoGEVEREIoGEVEREIoGEVEREIoGEVEREIoGEVEREL8/9YEgx2BYbfIAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 504x432 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# compare Fare with target\n",
    "plt.figure(figsize=(7,6))\n",
    "sns.distplot(df.Fare[df.Survived == 0], bins=25, color='r', label='Not Survived')\n",
    "sns.distplot(df.Fare[df.Survived == 1], bins=25, color='g', label='Survived')\n",
    "plt.legend();"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "40d6c741",
   "metadata": {},
   "source": [
    "## Categorical vs Target"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "9b87ca76",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "c:\\users\\imamx\\anaconda3\\envs\\midterm\\lib\\site-packages\\seaborn\\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.\n",
      "  warnings.warn(\n",
      "c:\\users\\imamx\\anaconda3\\envs\\midterm\\lib\\site-packages\\seaborn\\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.\n",
      "  warnings.warn(\n",
      "c:\\users\\imamx\\anaconda3\\envs\\midterm\\lib\\site-packages\\seaborn\\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.\n",
      "  warnings.warn(\n",
      "c:\\users\\imamx\\anaconda3\\envs\\midterm\\lib\\site-packages\\seaborn\\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.\n",
      "  warnings.warn(\n",
      "c:\\users\\imamx\\anaconda3\\envs\\midterm\\lib\\site-packages\\seaborn\\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "image/png": "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6yz23Xdf9t9/f+bOncsNN9ywzdtcunQpZ5999jhEBzvuuOO4bEeStGkeDyRpcHXtnsEkbwDur6rlSQ4dx+0uBBYCzJ49e7w2q3H0ta99jSuuuIKbbrqJ7bffngceeICf/vSnY1r38ccfZ+rU0f8t582bx7x588YzVElSF3k8kKTB1s0BZA4B5iU5ApgOPBM4F9glydTm6t8QsKapvwaYBaxOMhXYmc5AMr+kqhYBiwCGh4dr5LID3nXhVgW6/IMnbNV6Gt3atWvZbbfd2H777QHYbbfdAJgzZw7Lli1jt912Y9myZfzJn/wJ1113HWeccQbf+ta3uPvuu5k9ezb33HMP559/Pvvuuy8Ahx56KB/60Ie47bbbWLZsGWeddRb7778/99xzD0972tP48Y9/zAtf+ELuvvtuvvOd73DyySezbt06nv70p/OJT3yCF77whdxzzz383u/9Ho888gjz58/v23sjrbe17dV4su1Tt3k8kKTB1rVuolX1nqoaqqo5wLHANVX1+8C1wFFNtROBy5vppc08zfJrmmfoaIJ57Wtfy3333cfzn/983v72t/OlL31ps+usXLmSf/3Xf+Xiiy9mwYIFXHrppUDnRGLt2rUMDw8/WXfnnXdm7ty5T273iiuu4HWvex3Tpk1j4cKF/O3f/i3Lly/nQx/6EG9/+9sBOPXUU3nb297Grbfeyh577NGFvZYkbcjjgSQNtl6MJrqhdwPvSLKKzj2B5zfl5wO7NuXvAE7rQ2waBzvuuCPLly9n0aJFzJgxgwULFnDBBRdscp158+axww47AHDMMcdw2WWdMYYuvfRSjjrqqKfUX7BgAZdccgkAS5YsYcGCBTzyyCP8+7//O0cffTRz587lLW95C2vXrgXgq1/9Kscd13l+7/HHHz9euypJ2gSPB5I02Hry0Pmqug64rpm+GzholDqPAkf3Ih5135QpUzj00EM59NBD2W+//Vi8eDFTp07l5z/vjCW04ZDdz3jGM56cnjlzJrvuuiu33HILl1xyCR//+Mefsv158+bx3ve+l+9///ssX76cww47jB//+MfssssurFixYtSYHB1OknrP44EkDa5+XBnUJHfnnXdy1113PTm/YsUKnve85zFnzhyWL18OwGc/+9lNbmPBggX81V/9FQ8//DD777//U5bvuOOOHHjggZx66qm84Q1vYMqUKTzz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\n",
      "text/plain": [
       "<Figure size 1080x720 with 6 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "cat_var = ['Pclass', 'Sex', 'Parch', 'SibSp', 'Embarked']\n",
    "\n",
    "fig, axes = plt.subplots(2, 3, figsize=(15,10))\n",
    "for cat, ax in zip(cat_var, axes.flatten()):\n",
    "    sns.countplot(cat, data=df, hue='Survived', ax=ax)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fcb39197",
   "metadata": {},
   "source": [
    "## Feature Engineering"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "dbd30c94",
   "metadata": {},
   "source": [
    "### Adding feature --> Is Alone"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "3bf01902",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>IsAlone</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>PassengerId</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>S</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>S</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>S</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>S</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "             Survived  Pclass     Sex   Age  SibSp  Parch     Fare Embarked  \\\n",
       "PassengerId                                                                   \n",
       "1                   0       3    male  22.0      1      0   7.2500        S   \n",
       "2                   1       1  female  38.0      1      0  71.2833        C   \n",
       "3                   1       3  female  26.0      0      0   7.9250        S   \n",
       "4                   1       1  female  35.0      1      0  53.1000        S   \n",
       "5                   0       3    male  35.0      0      0   8.0500        S   \n",
       "\n",
       "             IsAlone  \n",
       "PassengerId           \n",
       "1              False  \n",
       "2              False  \n",
       "3               True  \n",
       "4              False  \n",
       "5               True  "
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['IsAlone'] = (df.SibSp == 0) & (df.Parch == 0)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ec0b7341",
   "metadata": {},
   "source": [
    "### Binning"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "850d4063",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>IsAlone</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>PassengerId</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>18-40</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>S</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>18-40</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>female</td>\n",
       "      <td>18-40</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>S</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>18-40</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>S</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>18-40</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>S</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "             Survived  Pclass     Sex    Age  SibSp  Parch     Fare Embarked  \\\n",
       "PassengerId                                                                    \n",
       "1                   0       3    male  18-40      1      0   7.2500        S   \n",
       "2                   1       1  female  18-40      1      0  71.2833        C   \n",
       "3                   1       3  female  18-40      0      0   7.9250        S   \n",
       "4                   1       1  female  18-40      1      0  53.1000        S   \n",
       "5                   0       3    male  18-40      0      0   8.0500        S   \n",
       "\n",
       "             IsAlone  \n",
       "PassengerId           \n",
       "1              False  \n",
       "2              False  \n",
       "3               True  \n",
       "4              False  \n",
       "5               True  "
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.Age = pd.cut(df.Age, [0,5,12,18,40,120], labels=['0-5', '5-12', '12-18', '18-40', '40-120'])\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "ee5a2f31",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>IsAlone</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>PassengerId</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>18-40</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0-25</td>\n",
       "      <td>S</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>18-40</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>25-100</td>\n",
       "      <td>C</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>female</td>\n",
       "      <td>18-40</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0-25</td>\n",
       "      <td>S</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>18-40</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>25-100</td>\n",
       "      <td>S</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>18-40</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0-25</td>\n",
       "      <td>S</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "             Survived  Pclass     Sex    Age  SibSp  Parch    Fare Embarked  \\\n",
       "PassengerId                                                                   \n",
       "1                   0       3    male  18-40      1      0    0-25        S   \n",
       "2                   1       1  female  18-40      1      0  25-100        C   \n",
       "3                   1       3  female  18-40      0      0    0-25        S   \n",
       "4                   1       1  female  18-40      1      0  25-100        S   \n",
       "5                   0       3    male  18-40      0      0    0-25        S   \n",
       "\n",
       "             IsAlone  \n",
       "PassengerId           \n",
       "1              False  \n",
       "2              False  \n",
       "3               True  \n",
       "4              False  \n",
       "5               True  "
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.Fare = pd.cut(df.Fare, [0,25,100,600], labels=['0-25', '25-100', '100-600'])\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9288dbbb",
   "metadata": {},
   "source": [
    "## Dataset Splitting"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "5c4cb57e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((712, 8), (179, 8), (712,), (179,))"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Dataset Splitting\n",
    "X = df.drop(columns='Survived')\n",
    "y = df.Survived\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)\n",
    "X_train.shape, X_test.shape, y_train.shape, y_test.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fb74b5b1",
   "metadata": {},
   "source": [
    "## Preprocessor"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "48e652a9",
   "metadata": {},
   "outputs": [],
   "source": [
    "numerical_pipeline = Pipeline([\n",
    "    ('imputer', SimpleImputer(strategy='mean')),\n",
    "    ('scaler', MinMaxScaler())\n",
    "])\n",
    "\n",
    "categorical_pipeline = Pipeline([\n",
    "    ('imputer', SimpleImputer(strategy='most_frequent')),\n",
    "    ('encode', OneHotEncoder())\n",
    "])\n",
    "\n",
    "preprocessor = ColumnTransformer([\n",
    "    ('numeric', numerical_pipeline,['SibSp','Parch']),\n",
    "    ('categoric', categorical_pipeline, ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'IsAlone'])\n",
    "])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3f82a64f",
   "metadata": {},
   "source": [
    "## Training"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "253f8a8e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Fitting 3 folds for each of 100 candidates, totalling 300 fits\n",
      "{'algo__n_neighbors': 7, 'algo__p': 2, 'algo__weights': 'uniform'}\n",
      "0.8328651685393258 0.8019477833327424 0.8156424581005587\n"
     ]
    }
   ],
   "source": [
    "pipeline = Pipeline([\n",
    "    ('prep', preprocessor),\n",
    "    ('algo', KNeighborsClassifier())\n",
    "])\n",
    "\n",
    "# Parameter tuning\n",
    "parameter = {\n",
    "    'algo__n_neighbors': range(1, 51, 2),\n",
    "    'algo__weights' : ['uniform', 'distance'],\n",
    "    'algo__p' : [1,2]\n",
    "}\n",
    "\n",
    "# Grid Search\n",
    "model = GridSearchCV(pipeline, parameter, cv=3, n_jobs=-1, verbose=1)\n",
    "model.fit(X_train, y_train)\n",
    "\n",
    "\n",
    "print(model.best_params_)\n",
    "print(model.score(X_train, y_train), model.best_score_, model.score(X_test, y_test))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "690aa18f",
   "metadata": {},
   "source": [
    "## Save and Load Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "ba58a5e0",
   "metadata": {},
   "outputs": [],
   "source": [
    "#save model\n",
    "filename = 'model_titanic_final.pkl'\n",
    "pickle.dump(model, open(filename, 'wb'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "b6da8410",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8156424581005587"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# load model\n",
    "#load model\n",
    "loaded_model = pickle.load(open(filename, 'rb'))\n",
    "results = loaded_model.score(X_test, y_test)\n",
    "results"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
