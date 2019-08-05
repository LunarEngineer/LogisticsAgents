import gym
from math import sqrt
import numpy as np

def getDesire(c):
  """
  This helper function takes a customer object and returns a numpy
  array containing the deficit by supply class for that customer.
  """
  desired_supply = np.full(c.supply_classes,c.supply_limit) - c.supplies
  return(desired_supply)
def getMission(trk,customer_needs,mission: int=None):
  """
  This helper function takes in a truck and a two-dimensional numpy
  array of customer needs and spits back a customer number to service
  in the range (0,Inf) where anything less than n dispatches to a
  customer, n to the depot, and anything greater than n results in
  no movement.
  """
  # This is necessary to input into the action space.
  pri = np.array(trk.supply_priority * 255, dtype='uint8')
  # The mission flag is passed to handle pre-existing missions
  if mission:
    cust = mission
  else:
    # Do an initial check to see if this truck needs to restock.
    if sum(trk.supplies)<1e-3:
      # If so, then set the customer to the depot.
      cust = customer_needs.shape[0]
    else:
      # Otherwise...
      # First, what needs can this truck service and how does that change
      # customer needs?
      # Second, what supplies does this truck have remaining? How does
      # *that* change the needs.
      # Finally, sum that up for each customer.    
      serviceable = np.sum(np.minimum(customer_needs*trk.allowed_supply,trk.supplies),1)
      # Here, things can branch.
      if sum(serviceable) < 1e-3:
        # If the truck cannot service *any* customer effectively
        #  either as a result of no customer *needing* anything
        #  or because the truck doesn't carry what the customer
        #  needs return an illegal customer number and the environment
        #  will interpret that as 'do not move.'
        cust = customer_needs.shape[0] + 1
      else:
        # Otherwise, out of the serviceable needs, which are highest?
        cust = np.argmax(serviceable)
  # Return the customer number and append the current supply priority.
  #  Note that this agent makes *no* attempt to modify supply priority.
  return(np.concatenate((np.atleast_1d(cust),pri)))

vfunc = np.vectorize(getDesire)
# This agent keeps track of customer needs and dispatches trucks accordingly.
env = gym.make('gym_logistics_simple:logistics-simple-v0')

# How many simulations am I going to run?
for i in range(1):
  env.reset()
  done = False
  # Initialize the dispatch status to 'no orders' for all trucks  
  # What trucks are available? this will set a row key for the
  # dispatch status matrix.
  trkInd = {k:t for k,t in zip([x for x in range(len(env.trucks))],[t for t in env.trucks])}
  # Initialize the customers
  customer_needs = np.vstack([getDesire(env.customers[c]) for c in env.customers])
  # Initialize all the truck orders.
  dispatch_status = np.full(env.action_space.shape,customer_needs.shape[0]+1,dtype='uint8')
  # Create a dummy first state
  s = env._next_observation()
  while not done:
    # Update the customers by calculating their deficit across supply.
    # Note that the agent will NOT have this, they will simply have remaining supply
    # and indirectly will have that relationship to max supply via the calculated
    # reward.
    customer_needs = np.vstack([getDesire(env.customers[c]) for c in env.customers])
    # Consider the trucks that are currently dispatched to service
    #  update the customer needs to reflect that they *will* be serviced
    #  and route trucks appropriately. Do a check here to unassign a truck
    #  if they've delivered
    for trk_i in range(len(dispatch_status)):
      trk = env.trucks[trkInd[trk_i]]
      # Update customer needs
      customer_number = dispatch_status[trk_i,0]

      if customer_number < customer_needs.shape[0]:
        # Is the truck currently at the customer?
        # Do a quick check for mission complete

        x_t,y_t = s[customer_number][0:2]
        x_0,y_0 = trk.location
        if sqrt((x_t-x_0)**2+(y_t-y_0)**2) < 1e-3:
          dispatch_status[trk_i][0] = customer_needs.shape[0] + 1
        else:
          newarr = np.maximum(0,customer_needs[customer_number]-trk.supplies)
          customer_needs[customer_number] = newarr
      elif customer_number == customer_needs.shape[0]:
          x_t,y_t = env.depot
          x_0,y_0 = trk.location
          if sqrt((x_t-x_0)**2+(y_t-y_0)**2) < 1e-3:
            dispatch_status[trk_i][0] = customer_needs.shape[0] + 1
    print("\t\tCUSTOMERNEEDS: {}".format(customer_needs))
    for trk_i in range(len(dispatch_status)):
      trk = env.trucks[trkInd[trk_i]]
      # Update customer needs
      customer_number = dispatch_status[trk_i,0]
      if customer_number > customer_needs.shape[0]:
        newarr = getMission(trk,customer_needs)
        dispatch_status[trk_i] = newarr
        customer_number = newarr[0]
        # Does this truck have a customer now?
        if customer_number < customer_needs.shape[0]:
          customer_needs[customer_number] = np.maximum(0,customer_needs[customer_number]-trk.supplies)
      else:
        dispatch_status[trk_i] = getMission(trk,customer_needs,customer_number)
    print("\t\tDISPATCH: {}".format(dispatch_status))
    # What actions do I need to take?
    s,r,done,info = env.step(dispatch_status)

    env.render()
env.close()
