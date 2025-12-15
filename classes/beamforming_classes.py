import numpy as np
from typing import List, Tuple, Optional, Dict


class Array:
    """Phased array class"""
    def __init__(self, name, array_type, num_elements, frequencies, steering_angle, position, 
                 meshgrid_x, meshgrid_y, beam_profile_x, beam_profile_y,
                 element_spacing=None, radius=None, arc_angle=None):
        self.name = name
        self.array_type = array_type
        self.number_of_elements = num_elements
        self.position = position
        self.frequencies = frequencies
        self.steering_angle = np.radians(steering_angle)
        
        # IMPORTANT: Force element_spacing to 0.5 for linear arrays to avoid grating lobes
        if array_type == "Linear":
            self.elements_spacing = 0.5  # Always use 0.5λ spacing
        else:
            self.elements_spacing = element_spacing if element_spacing is not None else 0.5
            
        self.radius = radius if radius is not None else 1.0
        self.arc_angle = arc_angle if arc_angle is not None else 120.0
        
        self.wavelengths = [1.0 / f if f != 0 else 1.0 / (f + 1e-12) for f in self.frequencies]
        self.meshgrid_x = meshgrid_x
        self.meshgrid_y = meshgrid_y
        self.beam_profile_x = beam_profile_x
        self.beam_profile_y = beam_profile_y
        
        # Initialize array data
        self.initialize_array_data()
    
    def initialize_array_data(self):
        """Initialize or re-initialize array data based on current parameters"""
        center_x, center_y = self.position
        
        if self.array_type == "Linear":
            # Linear array along x-axis with 0.5λ spacing
            element_spacing_value = 0.5 * self.wavelengths[0]  # FORCE 0.5λ
            
            # Create element positions relative to array center
            relative_positions = []
            for i in range(self.number_of_elements):
                x = i * element_spacing_value - (self.number_of_elements - 1) * element_spacing_value / 2
                relative_positions.append(np.array([x, 0.0]))
            
        elif self.array_type == "Curved":
            radius_value = self.radius if self.radius is not None else 1.0
            arc_angle_value = np.radians(self.arc_angle) if self.arc_angle is not None else np.radians(120.0)
            
            angles = np.linspace(-arc_angle_value / 2, arc_angle_value / 2, self.number_of_elements)
            relative_positions = []
            for angle in angles:
                x = radius_value * np.sin(angle)
                y = radius_value * np.cos(angle)
                relative_positions.append(np.array([x, y]))
        else:
            raise ValueError(f"Array type must be 'Linear' or 'Curved'")
        
        # Store relative positions (for phase calculations)
        self.unit_placements = relative_positions
        
        # Calculate absolute positions (for visualization)
        absolute_positions = [pos + np.array([center_x, center_y]) for pos in relative_positions]
        
        # Calculate distances to meshgrid points (using absolute positions)
        distances = []
        for abs_pos in absolute_positions:
            dist = np.sqrt((self.meshgrid_x - abs_pos[0])**2 + (self.meshgrid_y - abs_pos[1])**2)
            distances.append(dist)
        
        # Calculate distances to beam profile line
        beam_distances = []
        for abs_pos in absolute_positions:
            beam_dist = np.sqrt((self.beam_profile_x - abs_pos[0])**2 + (self.beam_profile_y - abs_pos[1])**2)
            beam_distances.append(beam_dist)
        
        # Calculate phase shifts for steering
        self.phase_shifts = self._calculate_phase_shifts()
        
        # Store array data
        self.array_data = {
            "positions": np.array(absolute_positions),
            "unit_placements": np.array(relative_positions),
            "distances": distances,
            "beam_distances": beam_distances,
            "phase_shifts": self.phase_shifts,
            "frequencies": self.frequencies,
            "wavelengths": self.wavelengths,
            "steering_angle": self.steering_angle,
            "array_type": self.array_type,
            "element_spacing": self.elements_spacing,
            "radius": self.radius,
            "arc_angle": self.arc_angle
        }
    
    def _calculate_phase_shifts(self):
        """Calculate phase shifts for beam steering
        Convention: 0° = broadside, positive angles steer in positive direction
        For linear array along x-axis: positive angle steers to +x direction
        """
        phase_shifts = []
        
        for elem_idx in range(self.number_of_elements):
            element_phases = []
            
            for wavelength in self.wavelengths:
                k = 2 * np.pi / wavelength
                
                # Get relative position
                rel_pos = self.unit_placements[elem_idx]
                x_rel = rel_pos[0]
                y_rel = rel_pos[1]
                
                # For linear array along x-axis: phase = -k * x * sin(θ_steer)
                # For general case: phase = -k * (x*sin(θ) + y*cos(θ))
                # But for steering we use the projection
                if self.array_type == "Linear":
                    # Linear array along x-axis
                    # θ_steer = 0° is broadside, +θ is to the right
                    phase = -k * x_rel * np.sin(self.steering_angle)
                else:
                    # Curved array - general formula
                    phase = -k * (x_rel * np.sin(self.steering_angle) + y_rel * np.cos(self.steering_angle))
                
                element_phases.append(phase)
            
            phase_shifts.append(element_phases)
        
        return phase_shifts
    
    def update_parameters(self, **kwargs):
        """Update array parameters and reinitialize"""
        if 'name' in kwargs:
            self.name = kwargs['name']
        if 'array_type' in kwargs:
            self.array_type = kwargs['array_type']
        if 'number_of_elements' in kwargs:
            self.number_of_elements = kwargs['number_of_elements']
        if 'frequencies' in kwargs:
            self.frequencies = kwargs['frequencies']
            self.wavelengths = [1.0 / f if f != 0 else 1.0 / (f + 1e-12) for f in self.frequencies]
        if 'steering_angle' in kwargs:
            self.steering_angle = np.radians(kwargs['steering_angle'])
        if 'position' in kwargs:
            self.position = kwargs['position']
        
        # Handle array-type specific parameters
        if 'element_spacing' in kwargs:
            if self.array_type == "Linear":
                self.elements_spacing = 0.5  # Force 0.5λ for linear
            else:
                self.elements_spacing = kwargs['element_spacing']
        if 'radius' in kwargs:
            self.radius = kwargs['radius']
        if 'arc_angle' in kwargs:
            self.arc_angle = kwargs['arc_angle']
        
        # Reinitialize with updated parameters
        self.initialize_array_data()
    
    def update_steering_angle(self):
        """Update phase shifts based on steering angle"""
        self.phase_shifts = self._calculate_phase_shifts()
        self.array_data["phase_shifts"] = self.phase_shifts
        self.array_data["steering_angle"] = self.steering_angle


arrays_scenarios = {}

def tumor_ablation(meshgrid_x, meshgrid_y, beam_profile_x, beam_profile_y):
    """Tumor ablation scenario"""
    arrays_scenarios.clear()
    
    bottom_array = Array("Bottom Array", "Linear", 8, [1], 0, [0, 10], 
                         meshgrid_x, meshgrid_y, beam_profile_x, beam_profile_y, 0.5)
    top_array = Array("Top Array", "Linear", 8, [1], 0, [0, 18], 
                      meshgrid_x, meshgrid_y, beam_profile_x, beam_profile_y, 0.5)
    left_array = Array("Left Array", "Linear", 8, [1], 90, [-8, 14], 
                       meshgrid_x, meshgrid_y, beam_profile_x, beam_profile_y, 0.5)
    right_array = Array("Right Array", "Linear", 8, [1], -90, [8, 14], 
                        meshgrid_x, meshgrid_y, beam_profile_x, beam_profile_y, 0.5)
    
    arrays_scenarios["Bottom Array"] = bottom_array
    arrays_scenarios["Top Array"] = top_array
    arrays_scenarios["Left Array"] = left_array
    arrays_scenarios["Right Array"] = right_array

def ultrasound(meshgrid_x, meshgrid_y, beam_profile_x, beam_profile_y):
    """Ultrasound scenario"""
    arrays_scenarios.clear()
    
    transducer_array = Array("Transducer", "Curved", 64, [1], 0, [0, 0], 
                             meshgrid_x, meshgrid_y, beam_profile_x, beam_profile_y, 
                             element_spacing=None, radius=5, arc_angle=60)
    arrays_scenarios["Transducer"] = transducer_array

def five_G(meshgrid_x, meshgrid_y, beam_profile_x, beam_profile_y):
    """5G scenario"""
    arrays_scenarios.clear()
    
    sender_array = Array("Sender Array", "Linear", 16, [1], 0, [0, 0], 
                         meshgrid_x, meshgrid_y, beam_profile_x, beam_profile_y, 0.5)
    arrays_scenarios["Sender Array"] = sender_array


class BeamformingSystem:
    """Main system class to manage arrays"""
    def __init__(self):
        self.arrays = {}
        
        # Mesh grid
        x = np.linspace(-20, 20, 100)
        y = np.linspace(0, 40, 100)
        self.meshgrid_x, self.meshgrid_y = np.meshgrid(x, y)
        
        # Beam profile line
        self.beam_profile_x = np.linspace(-20, 20, 500)
        self.beam_profile_y = 20 * np.ones_like(self.beam_profile_x)
        
        # Angles for polar plot (0° = broadside/up, clockwise)
        self.angles = np.linspace(0, 2 * np.pi, 360)
        self.polar_power = np.zeros_like(self.angles)
        
        self.current_array = None
        self.frame = 100
    
    def add_array(self, name, array_type, num_elements, frequencies, steering_angle, position,
                  element_spacing=None, radius=None, arc_angle=None):
        """Add a new array to the system"""
        if array_type == "Linear":
            element_spacing = 0.5  # Force 0.5λ
            radius = None
            arc_angle = None
        elif array_type == "Curved":
            if radius is None:
                radius = 1.0
            if arc_angle is None:
                arc_angle = 120.0
            element_spacing = None
        
        new_array = Array(name, array_type, num_elements, frequencies, steering_angle, position,
                          self.meshgrid_x, self.meshgrid_y, self.beam_profile_x, self.beam_profile_y,
                          element_spacing, radius, arc_angle)
        self.arrays[name] = new_array
        self.current_array = new_array
        return new_array
    
    def update_array(self, old_name, **kwargs):
        """Update an existing array"""
        if old_name in self.arrays:
            array = self.arrays[old_name]
            
            update_params = {}
            
            if 'name' in kwargs:
                update_params['name'] = kwargs['name']
            if 'array_type' in kwargs:
                update_params['array_type'] = kwargs['array_type']
            if 'number_of_elements' in kwargs:
                update_params['number_of_elements'] = kwargs['number_of_elements']
            if 'frequencies' in kwargs:
                update_params['frequencies'] = kwargs['frequencies']
            if 'steering_angle' in kwargs:
                update_params['steering_angle'] = kwargs['steering_angle']
            if 'position' in kwargs:
                update_params['position'] = kwargs['position']
            
            array_type = kwargs.get('array_type', array.array_type)
            
            if array_type == "Linear":
                update_params['element_spacing'] = 0.5  # Force 0.5λ
                update_params['radius'] = None
                update_params['arc_angle'] = None
            elif array_type == "Curved":
                radius = kwargs.get('radius', array.radius)
                arc_angle = kwargs.get('arc_angle', array.arc_angle)
                update_params['radius'] = radius if radius is not None else 1.0
                update_params['arc_angle'] = arc_angle if arc_angle is not None else 120.0
                update_params['element_spacing'] = None
            
            array.update_parameters(**update_params)
            
            if 'name' in update_params and update_params['name'] != old_name:
                new_name = update_params['name']
                self.arrays[new_name] = self.arrays.pop(old_name)
                return new_name
            return old_name
        return None
    
    def remove_array(self, name):
        """Remove an array from the system"""
        if name in self.arrays:
            del self.arrays[name]
            if self.current_array and self.current_array.name == name:
                self.current_array = None
    
    def get_array_info(self, name):
        """Get information about a specific array"""
        if name in self.arrays:
            array = self.arrays[name]
            return {
                'name': array.name,
                'type': array.array_type,
                'num_elements': array.number_of_elements,
                'frequencies': array.frequencies,
                'steering_angle': np.degrees(array.steering_angle),
                'position': array.position,
                'element_spacing': array.elements_spacing,
                'radius': array.radius,
                'arc_angle': array.arc_angle
            }
        return None
    
    def calculate_wave_field(self):
        """Calculate wave field and beam pattern - CORRECTED VERSION"""
        self.polar_power.fill(0)
        resultant_wave = np.zeros_like(self.meshgrid_x)
        beam_profile_amplitude = np.zeros_like(self.beam_profile_x)
        
        if not self.arrays:
            return {
                'wave_field': resultant_wave,
                'beam_profile': beam_profile_amplitude,
                'polar_power': self.polar_power,
                'frame': self.frame
            }
        
        for array in self.arrays.values():
            array_data = array.array_data
            unit_placements = array_data["unit_placements"]  # Relative positions
            distances = array_data["distances"]
            beam_distances = array_data["beam_distances"]
            phase_shifts = array_data["phase_shifts"]
            wavelengths = array_data["wavelengths"]
            steering_angle = array_data["steering_angle"]
            
            # Calculate wave field
            for elem_idx in range(len(distances)):
                for freq_idx, wavelength in enumerate(wavelengths):
                    k = 2 * np.pi / wavelength
                    phi = phase_shifts[elem_idx][freq_idx]
                    
                    r = distances[elem_idx]
                    beam_r = beam_distances[elem_idx]
                    
                    resultant_wave += np.sin(k * r + phi)
                    beam_profile_amplitude += np.sin(k * beam_r + phi)
            
            # Calculate beam pattern - CORRECTED FORMULA
            for freq_idx, wavelength in enumerate(wavelengths):
                k = 2 * np.pi / wavelength
                
                for theta_idx, theta in enumerate(self.angles):
                    # Convert polar angle to array coordinates
                    # theta = 0° is +Y (up), increases clockwise
                    # For horizontal linear array: we need the angle from broadside
                    
                    # Direction vector
                    dx = np.sin(theta)
                    dy = np.cos(theta)
                    
                    # Calculate array factor
                    af_complex = 0.0 + 0.0j
                    
                    for elem_idx in range(len(unit_placements)):
                        pos = unit_placements[elem_idx]
                        x_elem = pos[0]
                        y_elem = pos[1]
                        
                        # Array factor formula: exp(j * n * k * d * (sin(θ) - sin(θ_steer)))
                        # For linear array along x-axis
                        if array.array_type == "Linear":
                            # Observation phase
                            obs_phase = k * x_elem * dx
                        else:
                            # General case for curved arrays
                            obs_phase = k * (x_elem * dx + y_elem * dy)
                        
                        # Steering phase
                        steer_phase = phase_shifts[elem_idx][freq_idx]
                        
                        # Total phase = observation - steering compensation
                        # (steering phase already has negative sign)
                        total_phase = obs_phase + steer_phase
                        
                        # Add element contribution
                        af_complex += np.exp(1j * total_phase)
                    
                    # Power = |AF|^2
                    power = np.abs(af_complex) ** 2
                    self.polar_power[theta_idx] += power
        
        # Normalize and convert to dB
        max_power = np.max(self.polar_power)
        if max_power > 1e-10:
            normalized = self.polar_power / max_power
            polar_power_db = 10 * np.log10(normalized + 1e-10)
            polar_power_db = np.maximum(polar_power_db, -40)
        else:
            polar_power_db = -40 * np.ones_like(self.polar_power)
        
        return {
            'wave_field': resultant_wave,
            'beam_profile': beam_profile_amplitude,
            'polar_power': polar_power_db,
            'frame': self.frame
        }
    
    def get_arrays_positions(self):
        """Get positions of all array elements"""
        positions = []
        labels = []
        for array_name, array in self.arrays.items():
            array_positions = array.array_data["positions"]
            positions.extend(array_positions.tolist())
            labels.extend([f"{array_name}_{i}" for i in range(len(array_positions))])
        return positions, labels
    
    def apply_scenario(self, scenario_name):
        """Apply a predefined scenario"""
        self.arrays.clear()
        
        if scenario_name == "5G":
            five_G(self.meshgrid_x, self.meshgrid_y, self.beam_profile_x, self.beam_profile_y)
        elif scenario_name == "Ultrasound":
            ultrasound(self.meshgrid_x, self.meshgrid_y, self.beam_profile_x, self.beam_profile_y)
        elif scenario_name == "Tumor Ablation":
            tumor_ablation(self.meshgrid_x, self.meshgrid_y, self.beam_profile_x, self.beam_profile_y)
        else:
            return False
        
        self.arrays.update(arrays_scenarios)
        
        if self.arrays:
            self.current_array = list(self.arrays.values())[0]
        
        return True
    
    def clear_scenario(self):
        """Clear current scenario"""
        self.arrays.clear()
        self.current_array = None