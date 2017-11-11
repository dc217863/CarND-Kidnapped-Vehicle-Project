/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <limits>

#include "particle_filter.h"

using namespace std;

const double INITIAL_WEIGHT = 1.0;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    std::cout << "PF::Init" << std::endl;
    num_particles = 50;
    particles.resize(num_particles);
    weights.resize(num_particles);

    const double std_x = std[0];
    const double std_y = std[1];
    const double std_theta = std[2];

    std::default_random_engine gen;
    // normal guassian distribution
    std::normal_distribution<double> dist_x(x, std_x);
    std::normal_distribution<double> dist_y(y, std_y);
    std::normal_distribution<double> dist_theta(theta, std_theta);

    for (int i = 0; i < num_particles; ++i) {
        particles[i].id = i;
        particles[i].x = dist_x(gen);
        particles[i].y = dist_y(gen);
        particles[i].theta = dist_theta(gen);
        particles[i].weight = 1.0;

        weights[i] = 1.0;
    }
    is_initialized = true;
}


void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/
    std::cout << "PF::Prediction" << std::endl;
    std::default_random_engine gen;
    // normal guassian distribution with mean 0
    // NOTE: noise here should refer to measurement noise for velocity, yaw_rate. However, we use std_pos
    // for simplicity
    std::normal_distribution<double> dist_x(0.0, std_pos[0]);
    std::normal_distribution<double> dist_y(0.0, std_pos[1]);
    std::normal_distribution<double> dist_theta(0.0, std_pos[2]);

    for (auto& par : particles) {
        const double noise_x = dist_x(gen);
        const double noise_y = dist_y(gen);
        const double noise_theta = dist_theta(gen);

        if (abs(yaw_rate) > 0.001) {
            par.x += velocity / yaw_rate * (sin(par.theta + yaw_rate * delta_t) - sin(par.theta));
            par.y += velocity / yaw_rate * (cos(par.theta) - cos(par.theta + yaw_rate * delta_t));
            par.theta += yaw_rate * delta_t;
        }
        else {
            par.x += velocity * delta_t * cos(par.theta);
            par.y += velocity * delta_t * sin(par.theta);
            // theta remains the same as yaw_rate = 0
        }

        // add noise
        par.x += noise_x;
        par.y += noise_y;
        par.theta += noise_theta;        
    }
}


/**************************************************************
 * Find the predicted measurement that is closest to each observed measurement
 * and assign the observed measurement to this particular landmark.
 ***************************************************************/
void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, vector<LandmarkObs>& observations){

  for (int i = 0; i < observations.size(); i++) {

    int map_id;
    double current_smallest_error = std::numeric_limits<double>::max();

    for (int j = 0; j < predicted.size(); j++) {

      const double dx = predicted[j].x - observations[i].x;
      const double dy = predicted[j].y - observations[i].y;
      const double error = dx * dx + dy * dy;

      if (error < current_smallest_error) {
        map_id = j;
        current_smallest_error = error;
      }
    }
    observations[i].id = map_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
    const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// Updates the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
    //   http://planning.cs.uiuc.edu/node99.html
  // constants used later for calculating the new weights
  const double stdx = std_landmark[0];
  const double stdy = std_landmark[1];
  const double na = 0.5 / (stdx * stdx);
  const double nb = 0.5 / (stdy * stdy);
  const double d = sqrt( 2.0 * M_PI * stdx * stdy);

  for (int  i = 0; i < num_particles; i++) {

    const double px = particles[i].x;
    const double py = particles[i].y;
    const double ptheta = particles[i].theta;

    vector<LandmarkObs> landmarks_in_range;
    vector<LandmarkObs> map_observations;

    // Step 1: converting landmark observations from vehicle to map coordinates
    for (int j = 0; j < observations.size(); j++){

      const int oid = observations[j].id;
      const double ox = observations[j].x;
      const double oy = observations[j].y;

      const double transformed_x = px + ox * cos(ptheta) - oy * sin(ptheta);
      const double transformed_y = py + oy * cos(ptheta) + ox * sin(ptheta);

      LandmarkObs observation = {
        oid,
        transformed_x,
        transformed_y
      };

      map_observations.push_back(observation);
    }

    // Step 2: finding map landmarks within sensor range
    for (int j = 0;  j < map_landmarks.landmark_list.size(); j++) {

      const int mid = map_landmarks.landmark_list[j].id_i;
      const double mx = map_landmarks.landmark_list[j].x_f;
      const double my = map_landmarks.landmark_list[j].y_f;

      const double dx = mx - px;
      const double dy = my - py;
      const double error = sqrt(dx * dx + dy * dy);

      if (error < sensor_range) {

        LandmarkObs landmark_in_range = {
          mid,
          mx,
          my
         };

        landmarks_in_range.push_back(landmark_in_range);
      }
    }

   // Step 3: Find the predicted measurement that is closest to each observed 
        // measurement and assign the observed measurement to this particular landmark.
   this->dataAssociation(landmarks_in_range, map_observations);

    // Step 4: compare landmarks from map to the vehicle observations to update the weights
    double w = INITIAL_WEIGHT;

    for (int j = 0; j < map_observations.size(); j++){

      const int oid = map_observations[j].id;
      const double ox = map_observations[j].x;
      const double oy = map_observations[j].y;

      const double predicted_x = landmarks_in_range[oid].x;
      const double predicted_y = landmarks_in_range[oid].y;

      const double dx = ox - predicted_x;
      const double dy = oy - predicted_y;

      const double a = na * dx * dx;
      const double b = nb * dy * dy;
      const double r = exp(-(a + b)) / d;
      w *= r;
    }

    this->particles[i].weight = w;
    this->weights[i] = w;
  }
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  std::vector<Particle> resampled(num_particles);


  std::default_random_engine gen;
  std::discrete_distribution<int> index(weights.begin(), weights.end());

    for (unsigned int i=0; i<num_particles; i++) {
        resampled[i] = particles[index(gen)];
    }
    // replace old particles with resampled ones
    particles = resampled;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
