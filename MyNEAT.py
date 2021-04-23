import random, copy, enum
import numpy as np
import matplotlib.pyplot as plt

# TODO: Check if species is stagnant

nodeAddRate = 0.2
nodeDeleteRate = 0.2
connectionAddRate = 0.5
connectionDeleteRate = 0.5
connectionToggleRate = 0.01
distanceThreshold = 3
population = 20
generations = 10

connections = []
nodes = []
genomes = []
species = []
currentInnoNum = 0
currentNodeNum = 0

class SuperNEAT:
    def __init__(self):
        self.nodeAddRate = 0.2
        self.nodeDeleteRate = 0.2
        self.connectionAddRate = 0.5
        self.connectionDeleteRate = 0.5
        self.connectionToggleRate = 0.01
        self.distanceThreshold = 3
        self.population = 20
        self.generations = 1

    def updateSettings(self):
        # Run when settings are changed
        global nodeAddRate, nodeDeleteRate, connectionAddRate, connectionDeleteRate, connectionToggleRate, distanceThreshold, population, generations
        nodeAddRate = self.nodeAddRate
        nodeDeleteRate = self.nodeDeleteRate
        connectionAddRate = self.connectionAddRate
        connectionDeleteRate = self.connectionDeleteRate
        connectionToggleRate = self.connectionToggleRate
        distanceThreshold = self.distanceThreshold
        population = self.population
        generations = self.generations

    def setup(self, inputs, outputs):
        # Run at beginning
        self.con = SpeciesController()
        self.con.createInitialPopulation(inputs, outputs, 0)

    def update(self):
        
        # Run once at beginning of generation
        #self.con.removeUnderperformers()
        self.con.repopulate()
        self.con.resetFitnesses()
        self.con.clearSpecies()

        

        
        

        for genome in genomes:
            self.con.assignSpecies(genome)
            

        print("Species: ", len(species))

    def reset(self):
        # Run to reset everything
        global connections, nodes, genomes, species, currentInnoNum, currentNodeNum
        connections = []
        nodes = []
        genomes = []
        species = []
        currentInnoNum = 0
        currentNodeNum = 0


class TYPES(enum.Enum):
    Input = 0
    Output = 2
    Hidden = 1

class SuperNet:
    def __init__(self, genome):


        self.nodes = []
        self.connections = []

        for n in genome.nodes:
            newNode = Node(n.num, n.type, n.x)
            self.nodes.append(newNode)

        for c in genome.connections:
            newConnect = Connection(c.weight, c.enabled)
            for n in self.nodes:
                if n.num == c.fromNode.num:
                    newConnect.fromNode = n
                if n.num == c.toNode.num:
                    newConnect.toNode = n
            self.connections.append(newConnect)


        
        self.inputs = [node for node in self.nodes if node.type == TYPES.Input]
        self.hiddens = [node for node in self.nodes if node.type == TYPES.Hidden]
        self.outputs = [node for node in self.nodes if node.type == TYPES.Output]

        
        self.hiddens.sort(key=lambda x: x.x)

        for h in self.hiddens:
            h.connections = [con for con in self.connections if con.toNode.num == h.num]
        for o in self.outputs:
            o.connections = [con for con in self.connections if con.toNode.num == o.num]



    
    def calculate(self, inputs):
        if len(inputs) != len(self.inputs):
            print("Size of inputs does not match network")
            return

        for i in range(len(inputs)):
            self.inputs[i].output = inputs[i]

        for h in self.hiddens:
            h.calculate()

        for o in self.outputs:
            o.calculate()

        
        output = [o.output for o in self.outputs]
        #print(output)

        return output

def getAverageWeightDifference(g1, g2):

    innoNums1 = [con.innoNum for con in g1.connections]
    innoNums2 = [con.innoNum for con in g2.connections]
    weights1 = [con.weight for con in g1.connections]
    weights2 = [con.weight for con in g2.connections]
    highestInnoNum = max(max(innoNums1), max(innoNums2))


    matching = 0
    weightDiff = 0

    for i in range(highestInnoNum + 1):
        if i in innoNums1 and i in innoNums2:
            matching += 1
            weightDiff += abs(weights1[innoNums1.index(i)] - weights2[innoNums2.index(i)])

    if matching > 0:
        return weightDiff / matching
    else:
        return float("inf")

def countDisjointAndExcess(g1, g2):

    disjoint = 0
    excess = 0
    innoNums1 = [con.innoNum for con in g1.connections]
    innoNums2 = [con.innoNum for con in g2.connections]

    highestInnoNum = max(max(innoNums1), max(innoNums2))


    for i in range(highestInnoNum + 1):
        if i not in innoNums1 and i in innoNums2 and i < max(innoNums1):
            # Disjoint gene not in g1
            disjoint += 1
        elif i not in innoNums1 and i in innoNums2 and i >= max(innoNums1):
            # Excess gene not in g1
            excess += 1
        if i not in innoNums2 and i in innoNums1 and i < max(innoNums2):
            # Disjoint gene not in g2
            disjoint += 1
        elif i not in innoNums2 and i in innoNums1 and i >= max(innoNums2):
            # Excess gene not in g2
            excess += 1

    return (disjoint, excess)

def getDistance(g1, g2):

        c1 = 1.0
        c2 = 1.0
        c3 = 0.4
        N = 1

        if len(g1.connections) > len(g2.connections):
            N = len(g1.connections)
        else:
            N = len(g2.connections)

        D, E = countDisjointAndExcess(g1, g2)
        W = getAverageWeightDifference(g1, g2)


        distance = ((c1 * E) / N) + ((c2 * D) / N) + (c3 * W)

        #print(distance)

        return distance

def getAdjustedFitness(genome):
    adjustedFitness = genome.fitness / len(genome.species.genomes)

    return adjustedFitness

def newConnection(fromNode, toNode, enabled=True):
    global currentInnoNum, connections

    if fromNode.x >= toNode.x:
        print("Error: Connecting to previous node")
        return
    for i in connections:
        if fromNode.num == i.fromNode.num and toNode.num == i.toNode.num:
            # Connection already exists
            connection = ConnectionGene(fromNode, toNode, i.innoNum, enabled)
            return connection
    # Create new connection
    currentInnoNum += 1
    connection = ConnectionGene(fromNode, toNode, currentInnoNum, enabled)
    connections.append(connection)
    return connection

def newNode(type):
    global currentNodeNum, nodes

    currentNodeNum += 1

    node = NodeGene(currentNodeNum, type)
    nodes.append(node)
    return node

class Node:
    def __init__(self, num, type, x):
        self.num = num
        self.type = type
        self.output = 0
        self.x = x
        self.connections = []
        
    def calculate(self):
        s = 0
        for c in self.connections:
            if c.enabled:
                s += c.weight * c.fromNode.output

        self.output = self.activation(s)

    def activation(self, x):
        
        return 1 / (1 + np.exp(-x))

class Connection:
    def __init__(self, weight, enabled):
        self.fromNode = None
        self.toNode = None
        self.weight = weight
        self.enabled = enabled

class NodeGene:
    def __init__(self, num, type):
        self.num = num
        self.type = type
        self.output = 0
        self.x = 0



    def createCopy(self):
        newCopy = copy.copy(self)
        return newCopy



class ConnectionGene:
    def __init__(self, fromNode, toNode, innoNum, enabled):
        #self.weight = weight
        self.fromNode = fromNode
        self.toNode = toNode
        self.innoNum = innoNum
        self.weight = 0
        self.enabled = enabled

    def createCopy(self):
        newCopy = copy.copy(self)
        return newCopy

class Genome:
    def __init__(self):
        self.connections = []
        self.nodes = []
        self.fitness = 0
        self.species = None
        self.adjustedFitness = 0

    def createCopy(self):
        newCopy = copy.deepcopy(self)
        return newCopy

    def initialize(self, inputNum, outputNum, hiddenNum):
        input_nodes = []
        output_nodes = []
        for i in range(inputNum):
            node = newNode(TYPES.Input)
            self.nodes.append(node)
            input_nodes.append(node)
        for i in range(outputNum):
            node = newNode(TYPES.Output)
            node.x = 1
            self.nodes.append(node)
            output_nodes.append(node)

        for i in input_nodes:
            for j in output_nodes:
                connection = newConnection(i, j)
                self.connections.append(connection)

    def randomizeWeights(self):
        for c in self.connections:
            c.weight = random.uniform(-2.0, 2.0)

    def mutateNode(self):

        if (random.random() <= nodeAddRate):
            if len(self.connections) <= 0:
                return
            lastConnection = random.choice(self.connections)

            if lastConnection.fromNode.x < lastConnection.toNode.x:
                lastConnection.enabled = False

                fromNode1 = lastConnection.fromNode
                toNode1 = newNode(TYPES.Hidden)
                
                self.nodes.append(toNode1)

                fromNode2 = toNode1
                toNode2 = lastConnection.toNode

                toNode1.x = fromNode1.x + ((toNode2.x - fromNode1.x) / 2)

                connection1 = newConnection(fromNode1, toNode1)
                connection1.weight = 1
                connection2 = newConnection(fromNode2, toNode2)
                connection2.weight = lastConnection.weight

                self.connections.append(connection1)
                self.connections.append(connection2)


        if (random.random() <= nodeDeleteRate):
            timeout = 0
            node = random.choice(self.nodes)
            while node.type != TYPES.Hidden:
                node = random.choice(self.nodes)
                if timeout > 100:
                    return
                timeout += 1
            # TODO: Fix 
            found = True
            while found:
                for c in self.connections:
                    if c.fromNode.num == node.num or c.toNode.num == node.num:
                        self.connections.remove(c)
                        #deleteCount += 1
                for c in self.connections:
                    if c.fromNode.num == node.num or c.toNode.num == node.num:
                        found = True
                        break
                else:
                    found = False

                
                        #self.connections.remove(c)
                        #deleteCount += 1
                    #connections.remove(c)
                    
            

            self.nodes.remove(node)

            #print("Deleted node")

    def mutateConnection(self):
        if (random.random() <= connectionAddRate):
            selectNodes = random.sample(self.nodes, 2)
            timeout = 0
            
            while selectNodes[1].type == TYPES.Input or selectNodes[0].type == TYPES.Output or selectNodes[0].x >= selectNodes[1].x:
                selectNodes = random.sample(self.nodes, 2)
                if timeout > 100:
                    return
                timeout += 1
            connection = newConnection(selectNodes[0], selectNodes[1])

            innoNums = [con.innoNum for con in self.connections]

            if connection.innoNum not in innoNums:
                self.connections.append(connection)


        if (random.random() <= connectionDeleteRate):
            if len(self.connections) <= 1:
                return
            con = random.choice(self.connections)
            self.connections.remove(con)

    def mutateWeight(self):
        if len(self.connections) <= 0:
            return
        mutateType = random.random()
        c = random.choice(self.connections)

        if mutateType < 0.1:
            c.weight = c.weight * random.uniform(-2.0, 2.0)
        else:
            c.weight = random.uniform(-2.0, 2.0)

    def mutateToggleConnection(self):
        if len(self.connections) <= 1:
                return
        c = random.choice(self.connections)
        rand = random.random()

        if rand < connectionToggleRate:
            c.enabled = not c.enabled

    def drawNetwork(self):

        plt.title(str(self.species.id) + " - " + str(self.fitness))

        for node in self.nodes:
            if node.type == TYPES.Input:
                plt.plot([node.num**1/2], [node.x], 'ro-')
                plt.text(node.num**1/2, node.x, node.type.name + " " + str(node.x))
            elif node.type == TYPES.Hidden:
                plt.plot([node.num**1/2], [node.x], 'ro-')
                plt.text(node.num**1/2, node.x, node.type.name + " " + str(node.x))
            elif node.type == TYPES.Output:
                plt.plot([node.num**1/2], [node.x], 'ro-')
                plt.text(node.num**1/2, node.x, node.type.name + " " + str(node.x))
        
        for connection in self.connections:
            for d in [connection.fromNode, connection.toNode]:
                if connection.enabled == False:
                    plt.arrow(connection.fromNode.num**1/2, connection.fromNode.x, connection.toNode.num**1/2 - connection.fromNode.num**1/2, connection.toNode.x - connection.fromNode.x,
                    color='red', label=str(connection.weight))
                else:
                    plt.arrow(connection.fromNode.num**1/2, connection.fromNode.x, connection.toNode.num**1/2 - connection.fromNode.num**1/2, connection.toNode.x - connection.fromNode.x,
                    color='blue', shape='full', head_width=0.01, head_length=0.04, length_includes_head=True, label=str(connection.weight))
                    plt.annotate(str(connection.weight), (connection.fromNode.num**1/2 + (connection.toNode.num**1/2 - connection.fromNode.num**1/2)/2, connection.fromNode.x + ((connection.toNode.x - connection.fromNode.x) / 2)))


        plt.show()

class Species:
    def __init__(self):
        self.genomes = []
        self.removedGenomes = 0
        species.append(self)
        self.id = 0

    def getTotalFitness(self):
        totalFitness = 0
        for i in genomes:
            totalFitness += i.fitness

        return totalFitness

    def chooseRandomByFit(self):
        v = random.random() * self.getTotalFitness()
        c = 0
        for i in range(0, len(self.genomes)):
            c += self.genomes[i].fitness
            if (c > v):
                return self.genomes[i]
        return random.choice(self.genomes)

    def crossover(self, parent1, parent2):
        # Assuming parent1 is more fit
        offspring = Genome()
        # Copy nodes of more fit parent to offspring
        for node in parent1.nodes:
            offspring.nodes.append(node.createCopy())
        # Find genes with same innovation number and pick a random one for offspring
        # Add excess and disjoint genes of more fit parent to offspring and ignore for less fit parent
        innoNumParent1 = [c.innoNum for c in parent1.connections]
        innoNumParent2 = [c.innoNum for c in parent2.connections]
        diffInnoNums = list(set(innoNumParent1) - set(innoNumParent2))

        for c in parent1.connections:
            if c.innoNum in diffInnoNums:
                offspring.connections.append(c.createCopy())
            else:
                if random.random() > 0.5:
                    offspring.connections.append(c.createCopy())
                else:
                    offspring.connections.append(parent2.connections[innoNumParent2.index(c.innoNum)].createCopy())

        # for diff in diffInnoNums:
        #     offspring.connections.append(parent1.connections[innoNumParent1.index(diff)].createCopy())
        # for common in commonInnoNums:
        #     if random.random() > 0.5:
        #         offspring.connections.append(parent1.connections[innoNumParent1.index(common)].createCopy())
        #     else:
        #         offspring.connections.append(parent2.connections[innoNumParent2.index(common)].createCopy())

            




        # for i in parent1.connections:
            
        #     for j in parent2.connections:
        #         if i.innoNum == j.innoNum:
        #             randomParentConnection = random.choice([i, j])
        #             offspring.connections.append(randomParentConnection.createCopy())
        #             break
        #     else:
        #         offspring.connections.append(i.createCopy())


        return offspring

    
    def reproduce(self):
        parents = []

        for i in range(2):
            newParent = self.chooseRandomByFit()
            timeout = 0
            while newParent in parents:
                newParent = self.chooseRandomByFit()
                if timeout > 1000:
                    return
                timeout += 1
            parents.append(newParent)

        if parents[0].fitness >= parents[1].fitness:
            offspring = self.crossover(parents[0], parents[1])
        else:
            offspring = self.crossover(parents[1], parents[0])

        # Mutate offspring
        offspring.mutateNode()
        
        offspring.mutateConnection()
        
        offspring.mutateWeight()

        offspring.mutateToggleConnection()

        self.genomes.append(offspring)
        genomes.append(offspring)

class SpeciesController:
    def __init__(self):
        self.species = []

        self.currentId = 0

    def createInitialPopulation(self, inputNum, outputNum, hiddenNum):
        # TODO: Fix initial population (probably all referencing same thing)
        self.originalGenome = Genome()
        self.originalGenome.initialize(inputNum, outputNum, hiddenNum)
        #con.assignSpecies(originalGenome)

        for i in range(population):
            genome = self.originalGenome.createCopy()
            genome.randomizeWeights()
            genomes.append(genome)
            #con.assignSpecies(genome)
    def clearSpecies(self):
        self.species = []
        species.clear()
        self.currentId = 0

    def assignSpecies(self, genome):
        
        
        for s in self.species:
            if genome in s.genomes:
                s.genomes.remove(genome)
            if getDistance(s.genomes[0], genome) <= distanceThreshold:
                # Genome is part of species
                s.genomes.append(genome)
                genome.species = s
                return
        # Create new species and add genome
        newSpecies = Species()
        newSpecies.id = self.currentId
        self.currentId += 1
        newSpecies.genomes.append(genome)
        genome.species = newSpecies
        self.species.append(newSpecies)

    # def removeUnderperformers(self):
    #     #TODO: Also delete empty species

    #     for s in self.species:
    #         for g in s.genomes:
    #             g.adjustedFitness = getAdjustedFitness(g)

                
                

    #         s.genomes.sort(key=lambda g: g.adjustedFitness)

    #         for g in s.genomes[0:int(len(s.genomes)/2)]:
                
    #             s.genomes.remove(g)
    #             if g in genomes:
    #                 genomes.remove(g)
    #             else:
    #                 print("NOT IN GLOABL GENOME")

    #             s.removedGenomes += 1

    def repopulate(self):

        for s in self.species:
            for g in s.genomes:
                g.adjustedFitness = getAdjustedFitness(g)
            s.genomes.sort(key=lambda g: g.adjustedFitness)


            for g in s.genomes[0:int(len(s.genomes)/2)]:
                s.genomes.remove(g)
                genomes.remove(g)
                s.removedGenomes += 1


            while s.removedGenomes > 0:
                s.reproduce()
                s.removedGenomes -= 1

            for g in s.genomes:
                # Delete any genome with no connections
                if len(g.connections) <= 0:
                    s.genomes.remove(g)
                    genomes.remove(g)


        while len(genomes) < population:
            # TODO: Create copy probably uses same objects
            genome = self.originalGenome.createCopy()
            genomes.append(genome)
            

    def resetFitnesses(self):
        for genome in genomes:
            genome.fitness = 0
            genome.adjustedFitness = 0



class Sample:
    def __init__(self):
        self.onLeft = False
        self.goal = 10.0
        self.current = 0.0
        self.dead = False
        self.hit = 0
        self.totalhit = 0

    def calcFitness(self):
        fit = self.hit
        self.hit = 0
        return self.hit

    def update(self):
        if self.current < self.goal:
            self.onLeft = True
        else:
            self.onLeft = False

        if abs(self.current - self.goal) < 0.1:
            self.hit += 1
            self.totalhit += 1

    def move(self, dir):
        if dir == -1:
            self.current -= 1
        elif dir == 1:
            self.current += 1

    def reset(self):
        self.onLeft = False
        self.goal = 10.0
        self.current = 0.0
        self.dead = False
        self.hit = 0
        self.totalhit = 0


if __name__ == "__main__":

    neat = SuperNEAT()
    neat.setup(1, 1)



    isRandom = False

    for i in range(0, 300):

        neat.update()

        samples = []
        nets = []


        for x, genome in enumerate(genomes):
            #con.assignSpecies(genome)
            #nets.append(FeedForwardNetwork(genome))
            samples.append(Sample())
            nets.append(SuperNet(genome))

        print(i)

        #reset()
    
        
        for x, genome in enumerate(genomes):

            num = 0
            numHit = 0

            
            while True:


                samples[x].update()

                #net = SuperNet(genome)
                output = nets[x].calculate((samples[x].onLeft,))
                
                if not isRandom:
                    if output[0] < 0.5:
                        samples[x].move(-1)
                    else:
                        samples[x].move(1)
                else:
                    if random.random() < 0.5:
                        samples[x].move(-1)
                    else:
                        samples[x].move(1)
                
                genome.fitness += samples[x].calcFitness()**2


                print(samples[x].current)


                if samples[x].current < -10 or samples[x].current > 20:
                    samples[x].dead = True
                    samples[x].reset()
                    break

                numHit += samples[x].totalhit
                num += 1

                #if x == 0 and shown == False:

                  #  genome.drawNetwork()
                  #  shown = True
            
            avg = numHit / num
        print("Average: ", avg)
        print("Species: ", len(species))
        
        
        
    
    #genomes[0].drawNetwork()

       # genome.drawNetwork()

    
    print(len(genomes))
    for genome in genomes:

        net = SuperNet(genome)
        genome.drawNetwork()

        for c in genome.connections:
            if c.fromNode.x >= c.toNode.x:
                print(c)

