from SchedulingProblem import Job, Event, Resources, ProjectSchedulingModel, List
import numpy as np
import random
from Utils import topological_sort, plot_solution
from dataclasses import dataclass
import copy

@dataclass
class EventList:
    
    psmodel: ProjectSchedulingModel
    events: List[Event]
    jobs: List[Job]
    
    def __init__(self, psmodel: ProjectSchedulingModel, jobs: List[Job] | None = None):
        self.events = []
        self.psmodel = psmodel
        self.jobs = topological_sort(psmodel.jobs) if jobs is None else jobs
        for job in self.jobs:
            job.start_time = None
        self.create_event_list(psmodel.resources)
        
    def get_jobs(self) -> List[Job]:
        return [job for event in self.events for job in event.jobs]
    
    def get_makespan(self) -> int:
        return self.events[-1].startTime
    
    def plot(self, seed: int | None = None):
        plot_solution(self, seed=seed)
        
    def create_event_list(self, resources: Resources):
        
        self.add(self.jobs[0], start_time=0)
        current_time = 0
        for job in self.jobs[1:]:
            last_predecessor = max(filter(lambda j: j.id in job.predecessors, self.jobs), key=lambda j: j.start_time + j.duration)
            predecessors_end_time = last_predecessor.start_time + last_predecessor.duration
            base_time = max(predecessors_end_time, current_time)
            start_time = resources.get_time_for_resources(self.jobs, job, base_time)
            if start_time == current_time:
                self.add(job, event_id=-1)
            if start_time > current_time:
                self.add(job, start_time=start_time)
                current_time = start_time
            if start_time < current_time:
                raise ValueError("Invalid start time")
        
    def add(self, job: Job, start_time: int | None = None, event_id: int | None = None):
        if event_id is None and start_time is None:
            raise ValueError("Either start_time or event_id must be provided")
        if event_id is not None and start_time is not None:
            raise ValueError("Only one of start_time or event_id can be provided")
        if event_id and event_id >= len(self.events):
            raise ValueError("Event ID does not exist")
        if start_time and start_time < self.events[-1].startTime:
            raise ValueError("Start time must be after the last event")
        
        if event_id is not None:
            self.events[event_id].jobs.append(job)
            job.start_time = self.events[event_id].startTime
        if start_time is not None:
            self.events.append(Event(len(self.events), [job], start_time))
            job.start_time = start_time
                        
    def random_local_solution(self) -> 'EventList':
        
        new_jobs = copy.deepcopy(self.jobs)
        random_event = random.choice(self.events[:-1])
        
        for job in random_event.jobs:
            if job.id == 0:
                continue
            new_jobs.remove(job)
            inferior_pos = max(filter(lambda j: j.id in job.predecessors, new_jobs), key=lambda j: new_jobs.index(j))
            superior_pos = min(filter(lambda j: j.id in job.sucessors, new_jobs), key=lambda j: new_jobs.index(j))
            new_pos = random.randint(new_jobs.index(inferior_pos) + 1, new_jobs.index(superior_pos))
            # print(f"Moving job {job.id} to position {new_pos} in the interval {[new_jobs.index(inferior_pos) + 1, new_jobs.index(superior_pos)]}")
            # print(f"    Predecessors: {job.predecessors}")
            # print(f"    Sucessors: {job.sucessors}")
            new_jobs.insert(new_pos, job)
            
        return EventList(self.psmodel, new_jobs)
    
    def generate_new_local_solution(self, steps: int = 1) -> 'EventList':
        
        sol = self
        for _ in range(steps):
            sol = sol.random_local_solution()
            
        return sol

    def recombine_solution(self, other: 'EventList') -> 'EventList':
        self_events = []
        selected_jobs = []
        sorted_events = sorted(self.events, key=lambda e: len(e.jobs), reverse=True)
        for e in self.events:
            if e in sorted_events:
                self_events.append(copy.deepcopy(e))
                selected_jobs += e.jobs
            if len(selected_jobs) / 2 >= len(self.jobs):
                break
        other_jobs = copy.deepcopy(list(filter(lambda j: j not in selected_jobs, other.jobs)))
        selected_jobs = []
        while len(self_events) == 0 and len(other_jobs) == 0:
            j = other_jobs[0]
            if set(j.predecessors).issubset(selected_jobs):
                selected_jobs.append((other_jobs.pop(0)))
                continue
            selected_jobs += self_events.pop(0).jobs
            
            
        return None
    
    def generate_schedule_scheme(self) -> np.array:
        resources = self.psmodel.resources.resources
        end_time = self.events[-1].startTime
        schedules = [np.zeros((r, end_time), dtype=np.int32) - 1 for r in resources]
        jobs = [job for event in self.events for job in event.jobs]
        jobs = sorted(jobs, key=lambda x: x.duration, reverse=True)
        indexes = {i: {t: 0 for t in range(end_time)} for i in range(len(resources))}
        
        for job in jobs:
            for i, schedule in enumerate(schedules):
                
                # start = max((indexes[i][c] for c in range(job.start_time, job.start_time + job.duration)), default=0)
                # end = start + job.resources_needed[i]
                
                for col in range(end_time):
                    if job.isActiveAtTime(col):
                        start = indexes[i][col]
                        end = start + job.resources_needed[i]
                        schedule[start:end, col] = job.id
                        indexes[i][col] = end
                                                
        return schedules