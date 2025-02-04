
class SchedulingProblem:
    
    def __init__(self, nj, nr, j, r, sucs, ants):
        
        self.njobs = nj
        self.nresources = nr
        self.jobs = j
        self.resources = r
        self.succesors = sucs
        self.antecessors = ants
    
    @staticmethod
    def from_file(file_path):
        with open(file_path, "r") as file:
            lines = file.readlines()

        njobs = int(lines[5].split()[-1])
        renewable_resources_number = int(lines[8].split()[-2])
        nonrenewable_resources_number = int(lines[9].split()[-2])
        doble_constrain_resources_number = int(lines[10].split()[-2])
        nresources = renewable_resources_number + nonrenewable_resources_number + doble_constrain_resources_number
        
        succesors = []
        antecessors = [[] for i in range(njobs)]
        modes = []
        for i in range(18, 18 + njobs):
            line = lines[i].split()
            modes.append(int(line[1]))
            nsuccesor = int(line[2])
            it_succesors = [int(line[3 + j]) - 1 for j in range(nsuccesor)]
            succesors.append(it_succesors)
            for suc in it_succesors:
                antecessors[suc].append(i - 18)
                
        jobs = []
        for i in range(54, 54 + njobs):
            line = lines[i].split()
            jobs.append(Job(i - 54, modes[i - 54], int(line[2]), [int(line[3 + j]) for j in range(nresources)]))
            
        capacities_line = lines[89].split()
        resources = [Resource(i, int(capacities_line[i]), 'R') for i in range(nresources)]
        return SchedulingProblem(njobs, nresources, jobs, resources, succesors, antecessors)


class Job:
    def __init__(self, id, modes, duration, resources):
        self.id = id
        self.modes = modes
        self.duration = duration
        self.resources = resources
        
        
class Resource:
    def __init__(self, id, capacity, type):
        self.id = id
        self.capacity = capacity
        self.capacity_used = 0
        self.type = type