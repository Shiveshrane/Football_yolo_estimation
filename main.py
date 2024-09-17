from utils import read_video, save_video
from trackers import Tracker
import cv2
from team_assigner import TeamAssigner
import numpy as np
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance
def main():
    video_frames=read_video('input_vid\\08fd33_4.mp4')

    #shibesh
    #initialize tracker
    tracker=Tracker('models\\best.pt')
    tracks=tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')
    #get object positions
    tracker.add_position_to_tracks(tracks)

    #camera movement estimator
    camera_estimator=CameraEstimator(video_frames[0])
    camera_movement_per_frame=camera_estimator.get_camera_movement(video_frames, read_from_stub=True, stub_path='stubs/camera_movement.pkl')

    camera_estimator._adjust_position_to_tracks(tracks, camera_movement_per_frame)


    #view transformer
    view_transformer=ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    #interpolate ball positions
    tracks["ball"]=tracker.interpolate_ball_positions(tracks["ball"])


    #Speed and Distance etimator
    speed_and_distance_estimator=SpeedAndDistance()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)
    # #save cropped image of the player
    # for track_id, player in tracks['player'][0].items():
    #     bbox=player['bbox']
    #     frame=video_frames[0]
    #     cropped_image=frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]


    #     #save the cropped image
    #     cv2.imwrite('output_videos\\cropped_img.jpg', cropped_image)
    #     break
    


    #initialize team assigner
    team_assigner=TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['player'][0])

    for frame_num, player_track in enumerate(tracks['player']):
        for player_id, track in player_track.items():
            team=team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['player'][frame_num][player_id]['team']=team
            tracks['player'][frame_num][player_id]['team_color']=team_assigner.team_colors[team]

    #initialize player ball assigner
    player_ball_assigner=PlayerBallAssigner()
    team_ball_control=[]
    for frame_num, player_track in enumerate(tracks['player']):
        ball_bbox=tracks['ball'][frame_num][1]['bbox']
        assigned_player=player_ball_assigner.assign_ball_to_player(player_track, ball_bbox)
        if assigned_player!=-1:
            tracks['player'][frame_num][assigned_player]['has_ball']=True
            team_ball_control.append(tracks['player'][frame_num][assigned_player]['team'])
        else:
             team_ball_control.append(team_ball_control[-1])
    team_ball_control=np.array(team_ball_control)

             
    #Draw output 
    #Draw object tracks
    output_video_frames=tracker.draw_anotations(video_frames, tracks, team_ball_control)

    #draw camera movement
    output_video_frames=camera_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    #Draw speed and Distance
    output_video_frames=speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)
    #save video
    save_video(output_video_frames, 'output_videos\\output.avi')

if __name__ == "__main__":
    main()