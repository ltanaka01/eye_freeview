import os, glob, fnmatch, re
import numpy as np
import pandas as pd
from pathlib import Path

# subdivide a single .asc into a sequence of trials, there should be 85 per file
def get_trials(filename, ppd, xpixels, ypixels, sampling_rate):

	lines = open(filename).readlines()

	trials = []
	trial = []
	idat = []
	idats = []

	trial_started = False
	trial_ended = True

	for line in lines:

		if 'TRIALID' in line and trial_started == False and trial_ended == True:
			trial_started = True
			trial_ended = False

		elif 'TRIAL_RESULT' in line and trial_started == True and trial_ended == False:
			trial_started = False
			trial_ended = True
			trials.append(trial)
			
			# fix units of eye data
			# convert time to seconds
			# convert pixels to degrees
			if np.any(idat):
				idat = np.array(idat) # make array
				idat[:,0] = idat[:,0] - idat[0,0] # make first timepoint 0
				idat[:,0] = idat[:,0] / sampling_rate # convert to secondsd
				idat[:,1] = (idat[:,1] - xpixels/2) / ppd
				idat[:,2] = (idat[:,2] - ypixels/2) / ppd
			idats.append(np.array(idat))
			idat = []
			trial = []

		elif 'TRIALID' not in line and trial_started == True and trial_ended == False:
			if not line.split("\t")[0].isnumeric():
				trial.append(line)
			else:
				try:
					eye_t = np.int(line.split('\t')[0])
					eye_x = np.float(line.split('\t')[1])
					eye_y = np.float(line.split('\t')[2])
					idat.append([eye_t, eye_x, eye_y])
				except:
					pass

	return trials, idats

# hit/miss/abort and there are trials marked 'a'
def get_outcome(trial):

	# for each event in the trial
	for event in trial:

		# find the response and grab the type
		if 'TRIAL_VAR Response' in event:
			return event.split('TRIAL_VAR Response')[1].strip()

# time between go-signal and intial saccade onset
# go-signal for MGS (fixation offset) VGS/GAP (target presentation)
def get_latency(trial, task):

	for event in trial:

		# get target onset
		if task == 'MGS':
			if re.search(' FixationOff', event) or re.search(' FixationOff_T', event):
				target_offset = np.int(event.split('MSG')[1].split(' ')[0].strip())
				break
		elif task == 'VGS' or task == 'GAP':
			if re.search(' Target$', event):
				target_offset = np.int(event.split('MSG')[1].split(' ')[0].strip())
				break

	# get onset of the initial saccade
	saccade_onset = np.int(get_initial_saccade(trial).split("\t")[1])

	return saccade_onset-target_offset

def get_offset(trial, task):

	for event in trial:

		# get target onset
		if task == 'MGS':
			if re.search(' FixationOff', event) or re.search(' FixationOff_T', event):
				target_offset = np.int(event.split('MSG')[1].split(' ')[0].strip())
				break
		elif task == 'TSF':
			if re.search(' FixationOff\n', event) or re.search(' FixationOff_T', event):
				target_offset = np.int(event.split('MSG')[1].split(' ')[0].strip())
				break
		elif task == 'VGS' or task == 'GAP':
			if re.search(' Target$', event):
				target_offset = np.int(event.split('MSG')[1].split(' ')[0].strip())
				break
		else:
			target_offset = np.nan

	return target_offset


# get define the intial saccade as the one with the largest amplitude.
def get_initial_saccade(trial):

	amps = []
	events = []
	for event in trial:
		if 'ESACC' in event:
			amps.append(np.float(event.split('\t')[-2].strip()))
			events.append(event)

	return events[np.argmax(amps)]

def get_saccades(trial):
	saccades = []
	for event in trial:
		if 'ESACC' in event:
			saccades.append(event)

	return saccades


def get_accuracy(trial, ppd, xpixels, ypixels):

	for event in trial:

		# get target location
		if re.search(' targetlocation ', event):
			target_x = (np.int(event.split(' targetlocation [')[1].split(',')[0]) - xpixels) / ppd
			target_y = (np.int(event.split(' targetlocation [')[1].split(',')[1].split(']')[0]) - ypixels) / ppd
			break

	# get all saccades in a trial
	saccades = get_saccades(trial)

	# init output
	errors = []

	# for each saccade in the trial
	for saccade in saccades:

		# get location of the endpoint of the initial saccade
		initial_x = (np.float(saccade.split('\t')[5].strip()) - xpixels) / ppd
		initial_y = (np.float(saccade.split('\t')[6].strip()) - ypixels) / ppd

		# euclidean distance
		error = np.sqrt((target_x-initial_x)**2 + (target_y-initial_y)**2)

		# tally
		errors.append(error)

	return errors

def get_velocity(trial):
	
	# get duration of the initial saccade in seconds
	duration = np.int(get_initial_saccade(trial).split('\t')[2])/1000

	# get the amplitude of the intial saccade (in degrees already?)
	amplitude = np.float(get_initial_saccade(trial).split('\t')[-2])

	return amplitude/duration


def find_files(directory, pattern):
	names = []
	for root, _, files in os.walk(directory):
			for basename in files:
					if fnmatch.fnmatch(basename, pattern):
							filename = os.path.join(root, basename)
							names.append(Path(filename))
	return names

def get_target_data(trial, ppd, xpixels, ypixels):

	target_x, target_y, target_t, target2_x, target2_y, target2_t = [np.nan]*6
	
	for event in trial:
		if re.search(' targetlocation ', event):
			target_x = (np.int(event.split(' targetlocation [')[1].split(',')[0]) - xpixels) / ppd
			target_y = (np.int(event.split(' targetlocation [')[1].split(',')[1].split(']')[0]) - ypixels) / ppd
		if re.search(' First_Target_Presentation\n', event) or re.search(' Target\n', event):
			target_t = np.int(event.split(' ')[0].split('\t')[1])
			
		if re.search(' targetlocation2 ', event):
			target2_x = (np.int(event.split(' targetlocation2 [')[1].split(',')[0]) - xpixels) / ppd
			target2_y = (np.int(event.split(' targetlocation2 [')[1].split(',')[1].split(']')[0]) - ypixels) / ppd
		if re.search(' Target2\n', event):
			target2_t = np.int(event.split(' ')[0].split('\t')[1])

	return target_x, target_y, target_t, target2_x, target2_y, target2_t

def get_saccade_data(trial, task, ppd, xpixels, ypixels):

	# get saccades
	saccades = get_saccades(trial)

	# init output
	telemetry = []
	errors = []

	for saccade in saccades:

		try:

			saccade = re.sub('[^0-9\t\.]','',saccade).split('\t') 

			# get location+timing of the saccades
			saccade_x1 = (np.float(saccade[3]) - xpixels) / ppd
			saccade_y1 = (np.float(saccade[4]) - ypixels) / ppd
			saccade_t1 = np.int(saccade[0]) 

			saccade_x2 = (np.float(saccade[5]) - xpixels) / ppd
			saccade_y2 = (np.float(saccade[6]) - ypixels) / ppd
			saccade_t2 = np.int(saccade[1])

			amplitude = np.float(saccade[7])
			peak_velocity = np.int(saccade[8]) 

			duration = saccade_t2 - saccade_t1

			offset = get_offset(trial, task)

			# collate
			telemetry.append((saccade_x1, saccade_y1, saccade_t1, 
							  saccade_x2, saccade_y2, saccade_t2, 
							  duration, amplitude, peak_velocity))

			errors.append(np.nan)

		except Exception as e:
			errors.append(e)
			telemetry.append([np.nan]*10)

	return telemetry, errors


def run_analysis(task, ppd, xpixels, ypixels, sampling_rate, base_path):
	
	path = Path(base_path) / task
	
	# find all the files to run the analysis on
	files = find_files(path, '*.asc')

	# intialize output
	columns = ['task', 'group', 'subject', 'trial_number',
			   'saccade_x1', 'saccade_y1', 'saccade_t1',
			   'saccade_x2', 'saccade_y2', 'saccade_t2',
			   'saccade_number','duration', 'amplitude', 'peak_velocity', 'error']

	dfs = []
	bad_trials = []

	for file in files:
		print(file)

		# get the task, group, and subject
		group = file.as_posix().split('/')[1]
		subject = file.as_posix().split('/')[2]

		# get the trials
		trials, idats = get_trials(file, ppd, xpixels, ypixels, sampling_rate)

		# loop over trials and extract the data
		trial_num = 0
		for trial in trials:

			# counter
			trial_num += 1
							
			# get target info
			# target_x, target_y, target_t, target2_x, target2_y, target2_t  = get_target_data(trial, ppd, xpixels, ypixels)
				
			# get all saccades in a trial
			saccades, errors = get_saccade_data(trial, task, ppd, xpixels, ypixels)

			# loop over saccades
			saccade_num = 0
			for saccade,error in zip(saccades,errors):

				# counter
				saccade_num += 1

				# gather data
				data = [[ task, group, subject, trial_num,
						 saccade[0], saccade[1], saccade[2],
						 saccade[3], saccade[4], saccade[5],
						 saccade_num, saccade[6], saccade[7], saccade[8], error]]

				# create output
				df = pd.DataFrame(data = data, columns=columns)

				# store it
				dfs.append(df)

				# except Exception as e:
				# 	bad_trials.append([task,group,subject,trial[0], str(e)])


	return pd.concat(dfs)


# API
if __name__ == '__main__':

	# degrees of visual angle 
	# and other settings
	viewing_distance = 70 # cm
	xpixels = 1920/2 # pixels
	ypixels = 1080/2 # pixels
	pixels_per_cm = 37.79 # pixels/cm
	screen_width = xpixels/pixels_per_cm # cm
	ppd = np.pi*xpixels/np.arctan(screen_width/viewing_distance/2.0)/360.0 # pixels per degree
	sampling_rate = 500 # Hz
	base_path = Path('./')

	
	# analyze the REPEATED
	trials = run_analysis('REPEATED', ppd, xpixels, ypixels, sampling_rate, base_path)
	trials.to_csv('REPEATED_analyzed.csv', index=False)

	# analyze the NOVEL
	trials = run_analysis('NOVEL', ppd, xpixels, ypixels, sampling_rate, base_path)
	trials.to_csv('NOVEL_analyzed.csv', index=False)
	




