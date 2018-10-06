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

import java.util.*;

public class ReactiveTemplate implements ReactiveBehavior {

	private Random random;
	private double pPickup;
	private int numActions;
	private int numStates;
	private Agent myAgent;
	private int Best[];
	private double[] V;
	private List<City> city;
	double[][] R;
	double[][] T;
	final double NUM_TO_CONVERGE = 0.0001;


	@Override
	public void setup(Topology topology, TaskDistribution td, Agent agent) {

		// Reads the discount factor from the agents.xml file.
		// If the property is not present it defaults to 0.95
		Double discount = agent.readProperty("discount-factor", Double.class,
				0.95);

		this.random = new Random();
		this.pPickup = discount;
		System.out.println(discount);
		//this.pPickup = 0.5;
		this.numActions = topology.size()*2;
		this.myAgent = agent;
		this.numStates = topology.size();
		
		
		city = topology.cities();
		
		R = new double[numStates][numActions];
		
		for(int i = 0; i < numStates; i++) {
			for(int j = 0; j < numActions; j++) {
				if(j < numStates) {
					R[i][j] = new Double(td.reward(city.get(i), city.get(j)));	
				}
				else {
					R[i][j] = new Double((city.get(i).distanceTo(city.get(j-numStates))) * myAgent.vehicles().get(0).costPerKm() * (-1));
					//R[i][j] = 0;
				}
				R[i][j] *= NUM_TO_CONVERGE;
			}
		}
		
		T = new double[numStates][numActions];
		
		for(int i = 0; i < numStates; i++) {
			for(int j = 0; j < numActions; j++) {
				if(i == j) {
					T[i][j] = 0;
				}
				else if(j < numStates) {
					//System.out.println(td.probability(city.get(i), city.get(j)));
					T[i][j] = new Double(3*td.probability(city.get(i), city.get(j)));
				}
				else if(city.get(i).hasNeighbor(city.get(j-numStates))) {
					T[i][j] = new Double(td.probability(city.get(i), null)/city.get(i).neighbors().size());
				}
				else {
					T[i][j] = 0;
				}
				//System.out.println("T i:" + i  + "j: " + j + "value: " + T[i][j]);
			}
		}
		learnByReinforcement();
	}

	@Override
	public Action act(Vehicle vehicle, Task availableTask) {
		Action action; 
		int actIndex;
		City currentCity = vehicle.getCurrentCity();
		//if (availableTask == null || random.nextDouble() > pPickup) {
		if (availableTask == null) {
			actIndex = chooseBest(currentCity.id, -1);
			action = new Move(city.get(actIndex-numStates));
		} 
		else {
			actIndex = chooseBest(currentCity.id, availableTask.deliveryCity.id);
			if(actIndex < numStates) {
				action = new Pickup(availableTask);
			}
			else {
				action = new Move(city.get(actIndex - numStates));
			}
		}
		if (numActions >= 1) {
			System.out.println("The total profit after "+numActions+" actions is "+myAgent.getTotalProfit()+" (average profit: "+(myAgent.getTotalProfit() / (double)numActions)+")");
		}
		//numActions++;
		
		return action;
	}

	public void learnByReinforcement() {
		
		double threshold = 2000;
		boolean goodEnough = false; 
		boolean updateBelowThreshold = true;
		
		R = new double[numStates][numActions];
		double[][] Q = new double[numStates][numActions];
		
		V = new double[numStates];
		for(int i = 0; i < numStates; i++) {
			V[i] = (double)(random.nextInt(98999) + 1000);
			V[i] *= NUM_TO_CONVERGE;
		}
		while(!goodEnough) {
			updateBelowThreshold = true;
			for(int i = 0; i < numStates; i++) {
				for(int j = 0; j < numActions; j++) {
					double sum = 0;
					for(int k = 0; k < numStates; k++) {
						sum += T[i][k] * V[k];
					}
					Q[i][j] = R[i][j] + pPickup * sum;
				}
				double V_old = V[i];
				V[i] = Q[i][0];
				for(int j = 1; j < numActions; j++){
					V[i] = Math.max(V[i], Q[i][j]);
				}
				//System.out.println("Vi" + V[i]);
				//System.out.println(V_old);
				//System.out.println(Math.abs(V[i] - V_old));
				updateBelowThreshold = updateBelowThreshold && (Math.abs(V[i] - V_old) < threshold);
			} 
			goodEnough = updateBelowThreshold;
		}		
	}
	
	public int chooseBest(int s, int t) {
		double max;
		int maxIndex;
		//for(int i = 0; i < numStates; i++) {
			if(t == -1) {
				//System.out.println(V[0]);
				maxIndex = numStates;
				max = R[s][numStates] + pPickup * T[s][numStates] * V[0];
			}
			else {
				maxIndex = t;
				max = R[s][0] + pPickup * T[s][t] * V[t];
			}
			for(int j = numStates; j < numActions; j++) {
				int nextState = j - numStates;
				double cmp = R[s][j] + pPickup * T[s][j] * V[nextState];
				if(max < cmp) {
					max = cmp;
					maxIndex = j;
				}
			}
			//Best[i] = maxIndex;
		//}
			return maxIndex;
	}
}