use rand::Rng;
use std::collections::HashSet;
use std::fmt;

fn round_to(value: f64, decimal_places: u32) -> f64 {
    let multiplier = 10_f64.powi(decimal_places as i32);
    (value * multiplier).round() / multiplier
}

fn max_index(x: &Vec<f64>) -> usize {
    let mut max_val = x[0];
    let mut max_ind = 0;
    for i in 0..x.len() {
        if x[i] > max_val {
            max_val = x[i];
            max_ind = i;
        }
    }
    max_ind
}

fn index_of<T: PartialEq>(list: &[T], target: &T) -> Option<usize> {
    list.iter().position(|x| x == target)
}

fn action_formatted(x: Option<&Action>, weight: Option<&f64>) -> String {
    if let Some(action) = x {
        let action_text = match action {
            Action::Up => "↑",
            Action::Right => "→",
            Action::Down => "↓",
            Action::Left => "←"
        };
        let weight_format = if let Some(w) = weight {
            format!("{:.1}", w)
        } else {
            "_______".to_string()
        };
        format!("{}: {:<7}", action_text, weight_format)
    } else {
        "          ".to_string()
    }
}

#[derive(Clone, Debug, PartialEq, Copy)]
enum Action {
    Up,
    Right,
    Down,
    Left
}

#[derive(Clone, Debug)]
struct State {
    actions: Vec<Action>,
    action_values: Vec<f64>
}

impl State {
    fn policy (&self, epsilon: f64) -> Action {
        let mut rng = rand::rng();
        let random_number_1: f64 = rng.random::<f64>();

        if random_number_1 < epsilon {
            let random_index_2 = (rng.random::<f64>() * (self.actions.len() as f64)).floor() as usize;
            return self.actions[random_index_2];
        } else {
            return self.actions[max_index(&self.action_values)];
        }
    }
}

impl fmt::Display for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let acts: Vec<&str> = self.actions
            .iter()
            .map(|a| match a {
                Action::Up => "↑",
                Action::Right => "→",
                Action::Down => "↓",
                Action::Left => "←"
            })
            .collect();

        write!(
            f,
            "[<{}> | <{}>]",
            acts.join(""),
            self.action_values.iter().map(|x| format!("{:.1}", x)).collect::<Vec<_>>().join(" ")
        )
    }
}

#[derive(Clone)]
pub struct Board {
    data: Vec<Vec<State>>,
    dimensions: (usize, usize),
    start: (usize, usize),
    finish: (usize, usize),
    current: (usize, usize)
}

impl Board {
    pub fn new(rows: usize, columns: usize, start: (usize, usize), finish: (usize, usize), blocked: &HashSet<(usize, usize)>) -> Self {
        let mut data: Vec<Vec<State>> = Vec::new();
        for i in 0..rows {
            let mut temp: Vec<State> = Vec::new();
            for j in 0..columns {
                if blocked.contains(&(i + 1, j + 1)) {
                    temp.push(State {
                        actions: vec![],
                        action_values: vec![]
                    });
                    continue;
                }
                let mut actions: Vec<Action> = Vec::new();
                let mut action_values: Vec<f64> = Vec::new();
                if i >= 1 && !blocked.contains(&(i, j + 1)) {
                    actions.push(Action::Up);
                    action_values.push(0.0);
                }
                if j <= columns - 2 && !blocked.contains(&(i + 1, j + 2)) {
                    actions.push(Action::Right);
                    action_values.push(0.0);
                }
                if i <= rows - 2 && !blocked.contains(&(i + 2, j + 1)) {
                    actions.push(Action::Down);
                    action_values.push(0.0);
                }
                if j >= 1 && !blocked.contains(&(i + 1, j)) {
                    actions.push(Action::Left);
                    action_values.push(0.0);
                }
                temp.push(State {
                    actions: actions,
                    action_values: action_values
                });
            }
            data.push(temp);
        }
        Self {
            data: data,
            dimensions: (rows, columns),
            start: (start.0 - 1, start.1 - 1),
            finish: (finish.0 - 1, finish.1 - 1),
            current: (start.0 - 1, start.1 - 1)
        }
    }

    fn world_model(&mut self, a: &Action) -> f64 {
        match a {
            Action::Up => self.current = (self.current.0 - 1, self.current.1),
            Action::Right => self.current = (self.current.0, self.current.1 + 1),
            Action::Down => self.current = (self.current.0 + 1, self.current.1),
            Action::Left => self.current = (self.current.0, self.current.1 - 1)
        }
        if self.current == self.finish { 0.0 } else { -1.0 }
    }

    fn update_after_trajectory(&mut self, trajectory: &Vec<((usize, usize), Action, f64)>, discount_rate: f64, learning_rate: f64) {
        let mut returns: Vec<f64> = vec![0.0];
        for i in 1..trajectory.len() {
            returns.push(round_to(returns[i - 1] * discount_rate + (trajectory[trajectory.len() - i - 1].2 as f64), 5));
        }
        for i in 0..trajectory.len() {
            let current_traj = &trajectory[i];
            let current_state = &mut self.data[current_traj.0.0][current_traj.0.1];
            if let Some(index) = index_of(&current_state.actions, &current_traj.1) {
                current_state.action_values[index] += (returns[returns.len() - 1 - i] - current_state.action_values[index]) * learning_rate;
            }
        }
    }

    fn reset(&mut self) {
        self.current = self.start;
    }

    pub fn train(&mut self, num: u32, trajectory_limit: u32, discount_rate: f64, learning_rate: f64, epsilon: f64) {
        for _ in 0..num {
            let mut count = 0;
            let mut traj: Vec<((usize, usize), Action, f64)> = Vec::new();
            while self.current != self.finish && count < trajectory_limit {
                let current_state = &self.data[self.current.0][self.current.1];
                let curr = self.current;
                let action = current_state.policy(epsilon);
                let reward = self.world_model(&action);
                traj.push((curr, action, reward));
                count += 1;
            }
            self.update_after_trajectory(&traj, discount_rate, learning_rate);
            self.reset();
        }
    }

    pub fn trajectory(&mut self, trajectory_limit: u32, epsilon: f64) -> Vec<((usize, usize), (usize, usize))> {
        let mut final_vec: Vec<((usize, usize), (usize, usize))> = Vec::new();
        let mut count = 0;
        while self.current != self.finish && count < trajectory_limit {
            let _current_state = &self.data[self.current.0][self.current.1];
            let curr = self.current.clone();
            let _action = _current_state.policy(epsilon);
            let _reward = self.world_model(&_action);
            let next = self.current.clone();
            final_vec.push((curr, next));
            count += 1;
        }
        self.reset();
        //println!("{}", self);
        final_vec
    }
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "-")?;
        for _ in 0..self.dimensions.1 {
            write!(f, "-------------")?;
        }
        writeln!(f)?;
        for (m, row) in self.data.iter().enumerate() {
            for i in 0..4 {
                write!(f, "|")?;
                for (n, col) in row.iter().enumerate() {
                    let formatted = action_formatted(col.actions.get(i), col.action_values.get(i));
                    if (m, n) == self.start {
                        write!(f, " {}S|", formatted)?;
                    } else if (m, n) == self.finish {
                        write!(f, " {}F|", formatted)?;
                    } else {
                        write!(f, " {} |", formatted)?;
                    }
                }
                writeln!(f)?;
            }
            write!(f, "-")?;
            for _ in 0..self.dimensions.1 {
                write!(f, "-------------")?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}
