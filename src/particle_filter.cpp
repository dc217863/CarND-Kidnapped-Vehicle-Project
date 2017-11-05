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

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    std::cout << "PF::Init" << std::endl;
    num_particles = 50;
    particles.resize(num_particles);

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

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
    //   implement this method and use it as a helper during the updateWeights phase.
    // std::cout << "PF::DataAssociation" << std::endl;
    for (auto& obs : observations) {
        double current_error = std::numeric_limits<double>::max();
        int map_id = -1;
        for (const auto& pred : predicted) {
            const double dx = pred.x - obs.x;
            const double dy = pred.y - obs.y;
            const double error = dx*dx + dy*dy;
            if (error < current_error) {
                current_error = error;
                map_id = pred.id;
                // std::cout << "error: " << current_error << std::endl;
                // std::cout << pred.id << " pred.x: " << pred.x << ", pred.y: " << pred.y << std::endl;
            }
        }
        obs.id = map_id;
        // std::cout << "obs.x: " << obs.x << ", obs.y: " << obs.y << std::endl;
        // std::cout << "map_id: " << map_id << std::endl;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
    //   http://planning.cs.uiuc.edu/node99.html
    std::cout << "PF::UpdateWeights" << std::endl;
    for (auto& par : particles) {
        std::vector<LandmarkObs> map_observations;
        // Step 1: converting landmark observations from vehicle to map coordinates
        for (const auto& obs : observations) {
            double x_m = par.x + obs.x * cos(par.theta) - obs.y * sin(par.theta);
            double y_m = par.y + obs.x * sin(par.theta) + obs.y * cos(par.theta);
            map_observations.push_back(LandmarkObs{obs.id, x_m, y_m});
        }

        std::vector<LandmarkObs> landmarks_in_range;
        // Step 2: finding map landmarks within sensor range
        for (const auto& map_lm : map_landmarks.landmark_list) {
            const double dx = map_lm.x_f - par.x;
            const double dy = map_lm.y_f - par.y;
            const double error = sqrt(dx * dx + dy * dy);
            if (error < sensor_range) {
                landmarks_in_range.push_back(LandmarkObs{map_lm.id_i, map_lm.x_f, map_lm.y_f});
            }
        }

        // Step 3: Find the predicted measurement that is closest to each observed 
        // measurement and assign the observed measurement to this particular landmark.
        dataAssociation(landmarks_in_range, map_observations);

        // Step 4: compare landmarks from map to the vehicle observations to update the weights
        double s_x = std_landmark[0];
        double s_y = std_landmark[1];
        double lm_x, lm_y;
        for (const auto& obs : map_observations) {
            // get x,y for the landmark associated with the current observation
            for (const auto& lm : landmarks_in_range) {
                if (lm.id == obs.id) {
                    lm_x = lm.x;
                    lm_y = lm.y;
                }
            }
            const double k = (1/(2*M_PI*s_x*s_y));
            // calculate weight for this observation with multivariate Gaussian
            double obs_w = k * exp( -( pow(lm_x-obs.x,2)/(2*pow(s_x, 2)) + (pow(lm_y-obs.y,2)/(2*pow(s_y, 2))) ) );
            par.weight *= obs_w;
        }
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    std::cout << "PF::Resample" << std::endl;
    std::vector<Particle> resampled(num_particles);

    // make vector of all weights
    std::vector<double> weights;
    for (const auto& par : particles) {
        weights.push_back(par.weight);
    }

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
