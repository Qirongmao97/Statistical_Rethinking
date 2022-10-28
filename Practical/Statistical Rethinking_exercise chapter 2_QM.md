```python
import pymc as py
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
```

    WARNING (aesara.configdefaults): g++ not available, if using conda: `conda install m2w64-toolchain`
    WARNING (aesara.configdefaults): g++ not detected!  Aesara will be unable to compile C-implementations and will default to Python. Performance may be severely degraded. To remove this warning, set Aesara flags cxx to an empty string.
    

    Could not locate executable gfortran
    Could not locate executable f95
    Could not locate executable g95
    Could not locate executable efort
    Could not locate executable efc
    Could not locate executable flang
    don't know how to compile Fortran code on platform 'nt'
    

    WARNING (aesara.tensor.blas): Using NumPy C-API based implementation for BLAS functions.
    

### 2E1. Which of the expressions below correspond to the statement: the probability of rain on Monday? 

Answer: (2) by definition and (4) by Bayes' theorem

### 2E2. Which of the following statements corresponds to the expression: Pr(Monday | rain )?

Answer: (3)

### 2E3. Which of the expressions below correspond to the statement: the probability that it is Monday, given that it is raining?

Answer: (1) by definition and (4) by Bayes'theorem

### 2E4. The Bayesian statistician Bruno de Finetti (1906–1985) began his 1973 book on probability theory with the declaration: “PROBABILITY DOES NOT EXIST.” The capitals appeared in the original, so I imagine de Finetti wanted us to shout this statement. What he meant is that probability is a device for describing uncertainty from the perspective of an observer with limited knowledge; it has no objective reality. Discuss the globe tossing example from the chapter, in light of this statement. What does it mean to say “the probability of water is 0.7”?

Answer: I think the statement kind of makes sense."the probability of water is 0.7" is based on the observation from the prior (small world), and we knew a little about the "objective reality" (big world). But I would say the prior do have some connection with the reality, and as more observations in the prior, the connection gets bigger.

### 2M1. Recall the globe tossing model from the chapter. Compute and plot the grid approximate posterior distribution for each of the following sets of observations. In each case, assume a uniform prior for p. (1) W, W, W (2) W, W, W, L (3) L, W, W, L, W, W, W


```python
def posterior_grid_1(ngrid,water,trial):
    ## Define grid
    grid = np.linspace(0, 1, ngrid)
    ## Define prior
    prior = 1
    ## Binomial distribition
    likelihood = stats.binom.pmf(water,trial,grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior
```


```python
fig, axes = plt.subplots(figsize = (15, 4), nrows = 1, ncols = 3) 
## (1)
p_grid, posterior = posterior_grid_1(20, 3,3)
axes[0].plot(p_grid, posterior, 'o-')
axes[0].set_xlabel('Probability of water', fontsize=14)
axes[0].set_ylabel('Posterior probability', fontsize=14)
axes[0].set_title('{} points'.format(points))
## (2)
p_grid, posterior = posterior_grid_1(20, 3,4)
axes[1].plot(p_grid, posterior, 'o-')
axes[1].set_xlabel('Probability of water', fontsize=14)
axes[1].set_ylabel('Posterior probability', fontsize=14)
axes[1].set_title('{} points'.format(points))
## (3)
p_grid, posterior = posterior_grid_1(20, 5,7)
axes[2].plot(p_grid, posterior, 'o-')
axes[2].set_xlabel('Probability of water', fontsize=14)
axes[2].set_ylabel('Posterior probability', fontsize=14)
axes[2].set_title('{} points'.format(points))
```




    Text(0.5, 1.0, '20 points')




    
![png](output_11_1.png)
    


### 2M2. Now assume a prior for p that is equal to zero when p < 0.5 and is a positive constant when p ≥ 0.5. Again compute and plot the grid approximate posterior distribution for each of the sets of observations in the problem just above.


```python
def posterior_grid_2(ngrid,water,trial):
    ## Define grid
    grid = np.linspace(0, 1, ngrid)
    ## Define prior
    prior = (p_grid >= 0.5).astype(int)
    ## Binomial distribition
    likelihood = stats.binom.pmf(water,trial,grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior
```


```python
fig, axes = plt.subplots(figsize = (15, 4), nrows = 1, ncols = 3) 
## (1)
p_grid, posterior = posterior_grid_2(20, 3,3)
axes[0].plot(p_grid, posterior, 'o-')
axes[0].set_xlabel('Probability of water', fontsize=14)
axes[0].set_ylabel('Posterior probability', fontsize=14)
axes[0].set_title('{} points'.format(points))
## (2)
p_grid, posterior = posterior_grid_2(20, 3,4)
axes[1].plot(p_grid, posterior, 'o-')
axes[1].set_xlabel('Probability of water', fontsize=14)
axes[1].set_ylabel('Posterior probability', fontsize=14)
axes[1].set_title('{} points'.format(points))
## (3)
p_grid, posterior = posterior_grid_2(20, 5,7)
axes[2].plot(p_grid, posterior, 'o-')
axes[2].set_xlabel('Probability of water', fontsize=14)
axes[2].set_ylabel('Posterior probability', fontsize=14)
axes[2].set_title('{} points'.format(points))
```




    Text(0.5, 1.0, '20 points')




    
![png](output_14_1.png)
    


### 2M3. Suppose there are two globes, one for Earth and one for Mars. The Earth globe is 70% covered in water. The Mars globe is 100% land. Further suppose that one of these globes—you don’t know which—was tossed in the air and produced a “land” observation. Assume that each globe was equally likely to be tossed. Show that the posterior probability that the globe was the Earth, conditional on seeing “land” (Pr(Earth|land)), is 0.23.

P(Earth|land)= P(land|Earth)P(Earth)/P(land)

P(land|Earth)= 1-0.7=0.3

P(Earth)=0.5

P(land)=0.3*0.5+1*0.5=0.65

P(Earth|land)=0.3*0.5/0.65=0.23


### 2M4. Suppose you have a deck with only three cards. Each card has two sides, and each side is either black or white. One card has two black sides. The second card has one black and one white side. The third card has two white sides. Now suppose all three cards are placed in a bag and shuffled. Someone reaches into the bag and pulls out a card and places it flat on a table. A black side is shown facing up, but you don’t know the color of the side facing down. Show that the probability that the other side is also black is 2/3. Use the counting method (Section 2 of the chapter) to approach this problem. This means counting up the ways that each card could produce the observed data (a black side facing up on the table).

Card 1 : B B
    
Card 2 : B W
    
Card 3 : W W
    
Card 3 is impossible because both sides are white.

P(Black): P(BB)/P(BB)+P(BW)= 2/3

### 2M5. Now suppose there are four cards: B/B, B/W, W/W, and another B/B. Again suppose a card is drawn from the bag and a black side appears face up. Again calculate the probability that the other side is black.

P(Black):P(BB)/P(BB)+P(BW)=4/5

### 2M6. Imagine that black ink is heavy, and so cards with black sides are heavier than cards with white sides. As a result, it’s less likely that a card with black sides is pulled from the bag. So again assume there are three cards: B/B, B/W, and W/W. After experimenting a number of times, you conclude that for every way to pull the B/B card from the bag, there are 2 ways to pull the B/W card and 3 ways to pull the W/W card. Again suppose that a card is pulled and a black side appears face up. Show that the probability the other side is black is now 0.5. Use the counting method, as before.

Likelihood of Card 1: 1 * 2 = 2

Likelihood of Card 2: 2 * 1 = 2
    
Likelihood of Card 3: 3 * 0 = 0
    
P(Black): P(BB)/P(BB)+P(BW)= 2/ (2+2) = 0.5    

### 2M7. Assume again the original card problem, with a single card showing a black side face up. Before looking at the other side, we draw another card from the bag and lay it face up on the table. The face that is shown on the new card is white. Show that the probability that the first card, the one showing a black side, has black on its other side is now 0.75. Use the counting method, if you can. Hint: Treat this like the sequence of globe tosses, counting all the ways to see each observation, for each possible first card.

P(BB)=P(BB)/(P(BB)+P(BW))=6/8=0.75

### 2H1. Suppose there are two species of panda bear. Both are equally common in the wild and live in the same places. They look exactly alike and eat the same food, and there is yet no genetic assay capable of telling them apart. They differ however in their family sizes. Species A gives birth to twins 10% of the time, otherwise birthing a single infant. Species B births twins 20% of the time, otherwise birthing singleton infants. Assume these numbers are known with certainty, from many years of field research. Now suppose you are managing a captive panda breeding program. You have a new female panda of unknown species, and she has just given birth to twins. What is the probability that her next birth will also be twins?

Big world:

P(Twins|Species A)=0.1

P(Twins|Species B)=0.2

P(Species A)=P(Species B)=0.5

P(twins)= [P(Twins|Species A)*P(Species A)+P(Twins|Species B)*P(Species B)] =0.15

Small world:

P(A|twins)= P(Twins|Species A)*P(Species A)/P(twins)= 1/3

P(B|twins)= P(Twins|Species B)*P(Species B)/P(twins)= 2/3

##P(twins) here is prior
P(A)= P(A|twins)* P(twins)=1/3
P(B)= P(B|twins)* P(twins)=2/3

P(twins) = P(twins|A)*P(A) + P(twins|B)*P(B) = 1/3 * 1/10 + 2/3 * 2/10 = 1/6

### 2H2. Recall all the facts from the problem above. Now compute the probability that the panda we have is from species A, assuming we have observed only the first birth and that it was twins.

P(A|twins)= P(Twins|Species A)*P(Species A)/P(twins)= 1/3

### 2H3: Continuing on from the previous problem, suppose the same panda mother has a second birth and that it is not twins, but a singleton infant. Compute the posterior probability that this panda is species A

P(Twins,not Twins|Species A)=0.1 * 0.9 = 0.09

P(Twins,not Twins|Species B)=0.2 * 0.8 = 0.16

P(Species A)=P(Species B)=0.5

P(Species A|Twins,not Twins)= P(Twins,not Twins|Species A)* P(A)/ P(Twins, not Twins)= 0.09 * 0.05 / [(0.09+0.16)*0.05] = 9/25

### 2H4. A common boast of Bayesian statisticians is that Bayesian inference makes it easy to use all of the data, even if the data are of different types. So suppose now that a veterinarian comes along who has a new genetic test that she claims can identify the species of our mother panda. But the test, like all tests, is imperfect. This is the information you have about the test: • The probability it correctly identifies a species A panda is 0.8. • The probability it correctly identifies a species B panda is 0.65. The vet administers the test to your panda and tells you that the test is positive for species A. First ignore your previous information from the births and compute the posterior probability that your panda is species A. Then redo your calculation, now using the birth data as well.

Do they show seperate postive signal to A and B (A+,B+,A-,B-), or postive is A, negative is B?

P(A'|A) = 0.8

P(A'|B) = 1 - 0.65 = 0.35

P(A|A') = P(A'|A) * P(A) / [P(A'|A) * P(A) + P(A'|B) * P(B)] = 0.6956
  
  
Regarding the birth data, the likelihood of P(A) and P(B) are updated:
  
P(A)= 9/25 
  
P(B) = 1 - 9/25 = 16/25
  
P(A|A') = P(A'|A) * P(A) / [P(A'|A) * P(A) + P(A'|B) * P(B)] = [0.8 * 0.36] / [0.8 * 0.36 + 0.35 * 0.64] = 0.5625


```python

```
