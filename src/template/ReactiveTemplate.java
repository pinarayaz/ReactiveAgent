package template;

import java.util.Random;

import logist.simulation.Vehicle;
import logist.agent.Agent;
import logist.behavior.ReactiveBehavior;
import logist.plan.Action;
import logist.plan.Action.Move;
import logist.plan.Action.Pickup;
import logist.task.Task;
import logist.task.TaskDistribution;
import logist.topology.Topology;
import logist.topology.Topology.City;

import java.util.Arrays;
import java.util.List;

public class ReactiveTemplate implements ReactiveBehavior {

	private Random random;
	private double discountFactor;
	private int numActions;
	private int numStates;
	private Agent myAgent;
	private double maxReward, minReward;
	private double[] V;
	private List<City> city;
	double[][] R;
	double[][] T;



	@Override
	public void setup(Topology topology, TaskDistribution td, Agent agent) {

		// Reads the discount factor from the agents.xml file.
		// If the property is not present it defaults to 0.95
		Double discount = agent.readProperty("discount-factor", Double.class,
				0.95);
		this.random = new Random();
		this.discountFactor = 0.3; //= discount;
		this.numActions = topology.size()*2;
		this.myAgent = agent;
		this.numStates = topology.size();
		
		//EXPERIMENT
		System.out.println(discountFactor);
		
		city = topology.cities();
		
		//table for R(s,a)
		R = new double[numStates][numActions];
		
		this.maxReward = 0;
		this.minReward = 0;
		
		//Initializes R(s,a) table
		for(int i = 0; i < numStates; i++) {
			for(int j = 0; j < numActions; j++) {
				if(j < numStates)
					R[i][j] = new Double(td.reward(city.get(i), city.get(j)));
				else 
					R[i][j] = new Double((city.get(i).distanceTo(city.get(j-numStates))) * myAgent.vehicles().get(0).costPerKm() * (-1));
				
				//save max and min reward to initialize V values later
				if(R[i][j] > this.maxReward)
					this.maxReward = R[i][j];
				if(R[i][j] < this.minReward)
					this.minReward = R[i][j];
			}
		}
		
		//Table for T(s,a,s')
		T = new double[numStates][numActions];
		
		//Initializes T(s,a,s') table
		for(int i = 0; i < numStates; i++) {
			for(int j = 0; j < numActions; j++) {
				if(i == j || i == (j - numStates)) {
					T[i][j] = 0;
				}
				else if(j < numStates) {
					T[i][j] = new Double(td.probability(city.get(i), city.get(j)));
				}
				else if(city.get(i).hasNeighbor(city.get(j-numStates))) {
					T[i][j] = new Double(td.probability(city.get(i), null)/city.get(i).neighbors().size());
				}
				else {
					T[i][j] = 0;
				}
			}
		}
		
		//Use reinforcement learning to learn optimal actions at states
		try {
			learnByReinforcement();
		}catch(RuntimeException e) {
			System.out.println("Runtime exception, RLA did not converge.");
		}
	
	}

	@Override
	public Action act(Vehicle vehicle, Task availableTask) {
		Action action;
		
		//ID of action
		int actIndex;
		
		City currentCity = vehicle.getCurrentCity();
		
		//If a task is not available, choose best action among moving to different neighbor cities, 
		//else if a task is available, choose best action among picking up the available task, and refusing and moving to other neighbor cities
		if (availableTask == null) {
			actIndex = chooseBest(currentCity.id, -1);
			action = new Move(city.get(actIndex-numStates));
		} 
		else {
			actIndex = chooseBest(currentCity.id, availableTask.deliveryCity.id);
			if(isPickUp(actIndex))
				action = new Pickup(availableTask);
			else
				action = new Move(city.get(actIndex-numStates));
		}
		
		//In case of any action
		if (numActions >= 1) {
			System.out.println("The total profit of Agent: " + myAgent.id() + " after "+numActions+" actions is "+myAgent.getTotalProfit()+" (average profit: "+(myAgent.getTotalProfit() / (double)numActions)+")");
		}
		
		return action;
	}

	public void learnByReinforcement() {
		
		double threshold = 0.001; //threshold for convergence
		boolean goodEnough = false; 
		boolean updateBelowThreshold = true;

		
		double[][] Q = new double[numStates][numActions];
		
		//Initializes V
		V = new double[numStates];
		for(int i = 0; i < numStates; i++) {
			V[i] = (random.nextDouble() * (maxReward - minReward) + minReward);
		}
		
		//Value iteration
		while(!goodEnough) { //Until algorithm converges
			updateBelowThreshold = true;
			for(int i = 0; i < numStates; i++) {
				for(int j = 0; j < numActions; j++) {
					Q[i][j] = R[i][j] + discountFactor * T[i][j] * V[getNextState(j)];
				}
				double V_old = V[i];
				
				V[i] = Arrays.stream(Q[i]).max().getAsDouble();
			
				updateBelowThreshold = updateBelowThreshold && (Math.abs(V[i] - V_old) <= threshold);
			} 
			goodEnough = updateBelowThreshold;
		}		

	}
	
	//Choose best action (action with maximum reward) with given current state 's' and task 't', if no task is available t = -1
	public int chooseBest(int s, int t) {
		double max;
		int maxIndex;
		
		if(t == -1) { //If no task available, set max to reward of first action of movement to neighbor city. numStates is the starting index for actions of moving to neighbor cities
			maxIndex = numStates;
			max = R[s][numStates] + discountFactor * T[s][numStates] * V[getNextState(numStates)];
		}
		else { //If a task is available, set max to reward of the task
			maxIndex = t;
			max = R[s][t] + discountFactor * T[s][t] * V[getNextState(t)];
		}
		
		//Get the action with best reward
		for(int j = numStates; j < numActions; j++) { //Go through all options of neighbor cities
			double cmp = R[s][j] + discountFactor * T[s][j] * V[getNextState(j)];
			if(max < cmp) {
				max = cmp;
				maxIndex = j;
			}
		}
		
		return maxIndex;
	}
	
	//Is the action 'a' a pick up action?
	public boolean isPickUp(int a) {
		return a < numStates;
	}
	
	//Get the next state after action 'a'
	public int getNextState(int a) {
		return isPickUp(a) ? a : a - numStates;
	}
	
}